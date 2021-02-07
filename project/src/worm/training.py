
from abc import abstractclassmethod
from os import close, stat
from sys import platform
import json
from uuid import uuid4
from collections import defaultdict
from pathlib import Path
import itertools
import numpy as np
import torch

from ..agents.agent import Agent
from ..agents.a2c import A2CLearner
from ..agents.ppo_v2 import PPOv2Learner
from ..agents.randomagent import RandomAgent
from ..exec.executor import Runner, Executor
from .domain import WormDomain
from ..utils.color import bcolors

import mlflow

DEFAULT_EPISODES = 5000
DEFAULT_SAVE_FROM_EPISODE = 1000
DEFAULT_RESULT_DIR = "debug"

class AgentRunner(Runner):

    @staticmethod
    def add_parse_args(parser):
      parser.add_argument('-n','--episodes', type=int, default=DEFAULT_EPISODES, help='training episodes')
      parser.add_argument('-s','--save_from_episode', type=int, default=DEFAULT_SAVE_FROM_EPISODE, help='every x the agent state is safed to disk')
      parser.add_argument('-r', '--result_dir', type=str, default=DEFAULT_RESULT_DIR, help='result directory')
      parser.add_argument('-c', '--continue_training', default=False, action='store_true')
      parser.add_argument('-sd', '--state_dict', type=str, default=None,)
      parser.add_argument('-u', '--upload_state_dicts', default=False, action='store_true')
      return parser

    def __init__(self,
        episodes=DEFAULT_EPISODES,
        result_dir = DEFAULT_RESULT_DIR,
        save_from_episode = DEFAULT_SAVE_FROM_EPISODE,
        upload_state_dicts = False,
        **kwargs):
        super().__init__()
        self.training_episodes = episodes
        self.save_from_episode = save_from_episode
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.upload_state_dicts = upload_state_dicts

    @staticmethod
    def print_stats_on_console(nr_episode, sum_reward):
        if nr_episode % 1 == 0:
            print("e:", nr_episode , ", r:", sum_reward)
    
    @staticmethod
    def train_buffer(domain, env, agent, step_counter, nr_episode=0):
        state = env.reset()
        sum_reward = 0
        done = False
        measures_dict = None
        states=[]
        prob, value = None, None
        while not done:
            
            # 1. Select action according to policy
            action, prob, value = agent.policy(state)

            # 2. Execute selected action
            try:
                next_state, reward, done, _ = env.step(action) # _ = decision_steps but are not used here
                agent.remember(state, action, prob, value, reward, done)
                step_counter += 1
            except Exception as e:
                raise Exception(f"Stepping env failed:\nstate:{state}\naction:{action}") from e

            # 3. Integrate new experience into agent
            if step_counter % agent.get_buffersize() == 0:
                agent.update()
            
            # 4 step through
            state = next_state
            states.append(state)
            sum_reward += reward

        measures_dict = agent.get_measures()
        AgentRunner.print_stats_on_console(nr_episode, sum_reward)

        # Add domain specific measures
        return {
            **domain.evaluate(states),
            **measures_dict,
            "reward": sum_reward,
            "episode": nr_episode,
        }, step_counter

    @staticmethod
    def train_episode(domain, env, agent, nr_episode=0):
        state = env.reset()
        sum_reward = 0
        done = False
        measures_dict = None
        states=[]
        while not done:

            # 1. Select action according to policy
            action = agent.policy(state)

            # 2. Execute selected action
            try:
                next_state, reward, done, _ = env.step(action) # _ = decision_steps but are not used here
            except Exception as e:
                raise Exception(f"Stepping env failed:\nstate:{state}\naction:{action}") from e

            # 3. Integrate new experience into agent
            measures_dict = agent.update(nr_episode, state, action, reward, next_state, done)

            # 4 step through
            state = next_state
            states.append(state)
            sum_reward += reward
        
        AgentRunner.print_stats_on_console(nr_episode, sum_reward)

        # Add domain specific measures
        return {
            **domain.evaluate(states),
            **measures_dict,
            "reward": sum_reward,
            "episode": nr_episode,
        }

    def group_measures_by_keys(self, measure_dicts):
        def flatmap(func, *iterable):
            return itertools.chain.from_iterable(map(func, *iterable))
        squeezed_measurements_dicts = defaultdict(list)
        for key, value in flatmap(lambda d: d.items(), measure_dicts):
            squeezed_measurements_dicts[key].append(value)
      
    def run(self,
        worker_id : int,
        run_id : str,
        agent_class : Agent,
        state_dict=None,
        continue_training=False,
        **kwargs):

      mlflow.set_experiment(str(self.result_dir))
      with mlflow.start_run():
        mlflow.log_param("local_run_id", run_id)
        """ Loading old state dict """
        if state_dict:
            state_dict = torch.load(state_dict)


        domain = WormDomain(**kwargs)
        env = domain.create_env(worker_id, **kwargs)

        """Agent setup"""
        agent = agent_class(
            env=env,
            only_model=  not continue_training,
            state_dict = state_dict,
            **kwargs
        )

        """Initialize Training"""
        results = []
        start_episode = 0
        if continue_training:
            print("Loading old training state...")
            #results = state_dict["trainer"]["results"]
            start_episode = int(state_dict["trainer"]["cur_episode"]) + 1
            print("Loaded old training state.")

        def save_agent_state(episode,):
            print("Saving agent state...")
            save_path = self.result_dir / "{}.state_dict".format(run_id)
            torch.save({
                "agent": agent.state_dict(),
                "trainer":{
                    #"results": results,
                    "cur_episode": episode,
                }
            }, save_path)
            print("Saved agent state.")
            if self.upload_state_dicts:
              print("Uploading agent state...")
              mlflow.log_artifact(str(save_path))
              print("Uploaded agent state.")


        rewards = []
        best_avg_reward = 0
        agent_is_ppo = True if isinstance(agent, PPOv2Learner) else False
        step_counter = 0

        """Training"""
        for i in range(start_episode, self.training_episodes):

            if agent_is_ppo:
                measures, step_counter = self.train_buffer(domain, env, agent, step_counter, nr_episode=i) # --> None
            else:
                measures = self.train_episode(domain, env, agent, nr_episode=i)

            rewards.append(measures["reward"])
            avg_reward = np.mean(rewards[-100:])
            measures["reward_avg"] = avg_reward
            if avg_reward > best_avg_reward and i >= self.save_from_episode:
                best_avg_reward = avg_reward
                print(f"{bcolors.FAIL}Yay, new best average: {best_avg_reward}{bcolors.ENDC}")
                if len(rewards) >= 100:
                    save_agent_state(i)
            mlflow.log_metrics(measures, step=i)
            results.append(measures)

        env.close()

        return self.group_measures_by_keys(results)
