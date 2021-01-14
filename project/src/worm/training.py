
from abc import abstractclassmethod
from os import close, stat
from sys import platform
import json
from uuid import uuid4
from collections import defaultdict
from pathlib import Path
import itertools
import torch


from ..agents.agent import Agent
from ..agents.a2c import A2CLearner
from ..agents.randomagent import RandomAgent
from ..exec.executor import Runner, Executor
from .domain import WormDomain

import mlflow

DEFAULT_EPISODES = 5000
DEFAULT_SAVE_INTERVAL = 1000
DEFAULT_RESULT_DIR = "debug"

class AgentRunner(Runner):

    @staticmethod
    def add_parse_args(parser):
      parser.add_argument('-n','--episodes', type=int, default=DEFAULT_EPISODES, help='training episodes')
      parser.add_argument('-s','--save_interval', type=int, default=DEFAULT_SAVE_INTERVAL, help='every x the agent state is safed to disk')
      parser.add_argument('-r', '--result_dir', type=str, default=DEFAULT_RESULT_DIR, help='result directory')
      parser.add_argument('-c', '--continue_training', default=False, action='store_true')
      parser.add_argument('-sd', '--state_dict', type=str, default=None,)
      return parser

    def __init__(self,
        episodes=DEFAULT_EPISODES,
        result_dir = DEFAULT_RESULT_DIR,
        save_interval = DEFAULT_SAVE_INTERVAL,
        **kwargs):
        super().__init__()
        self.training_episodes = episodes
        self.save_interval = save_interval
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def train_episode(domain, env, agent, nr_episode=0):
        state = env.reset()
        sum_reward = 0
        done = False
        time_step = 0
        measures_dict = None
        states=[]
        while not done:
            # 1. Select action according to policy
            action = agent.policy(state)
            # 2. Execute selected action
            next_state, reward, done, _ = env.step(action) # _ = decision_steps but are not used here
            # 3. Integrate new experience into agent
            measures_dict = agent.update(nr_episode, state, action, reward, next_state, done)
            # 4 step through
            state = next_state
            states.append(state)
            sum_reward += reward
            time_step += 1
        if nr_episode % 25 == 0:
            print(f"e: {nr_episode}, r: {sum_reward}, i: {time_step}")
        # Add domain specific measures
        return {
            **domain.evaluate(states),
            **measures_dict,
            "reward": sum_reward,
            "episode":nr_episode,
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
            results = state_dict["trainer"]["results"]
            start_episode = int(state_dict["trainer"]["cur_episode"]) + 1
            print("Loaded old training state.")

        def save_agent_state(episode, results):
            print("Saving agent state...")
            save_path = self.result_dir / "{}_episode_{:02d}.state_dict".format(run_id, i)
            torch.save({
                "agent": agent.state_dict(),
                "trainer":{
                    "results": results,
                    "cur_episode": episode,
                }
            }, save_path)
            print("Saved agent state.")


        """Training"""
        for i in range(start_episode, self.training_episodes):
             measures = self.train_episode(domain, env, agent, nr_episode=i)
             results.append(measures)
             mlflow.log_metrics(measures, step=i)
             if (i+1)% self.save_interval == 0:
                 save_agent_state(i, results)

        save_agent_state(i, results)
        env.close()
        return self.group_measures_by_keys(results)