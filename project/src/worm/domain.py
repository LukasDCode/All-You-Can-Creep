
from os import close, stat
from sys import platform
import json
from uuid import uuid4
from collections import defaultdict
from pathlib import Path
import itertools
import torch

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import mlflow

from ..agents.a2c import A2CLearner
from ..agents.randomagent import RandomAgent
from ..exec.executor import Domain as DomainTrainingAdaptor, Executor 


class WormDomainAdaptor(DomainTrainingAdaptor):

    @staticmethod
    def add_parse_args(parser):
      parser.add_argument('-n','--episodes', type=int, default=2000, help='training episodes')
      parser.add_argument('-s','--save_interval', type=int, default=10, help='every x the agent state is safed to disk')
      parser.add_argument('-v', '--visualize', type=bool, default=False, help='call env.render')
      parser.add_argument('-t', '--time_scale', type=float, default=20, help='simulation speed scaling, higher values make physics less precise')
      parser.add_argument('-r', '--result_dir', type=str, default="result", help='result directory')
      return parser


    def __init__(self, config):
        super().__init__()
        self.render_env = config.visualize
        self.training_episodes = config.episodes
        self.result_dir = Path(config.result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.time_scale = config.time_scale
        self.save_interval = config.save_interval
        # self.distances = []


    def run(self, worker_id, run_id, params, state_dict=None, continue_training=False, **kwds):
        (rewards, measurements_dicts) = self.run_with_params(
            run_id=run_id,
            worker_id=worker_id,
            params=params,
            state_dict=state_dict,
            continue_training= continue_training,
        )
        result_dump = {
            "algorithm": "a2c",
            "params": params,
            "measures": {"rewards": rewards, **measurements_dicts}
        }
        result_path = self.result_dir / (run_id + ".results.json")
        with result_path.open('w+') as file:
            json.dump(result_dump, file, indent=2)
        return rewards

    def param_dict(self):
        return [
            ("alpha", 0.001, 0.001),
            ("gamma", 0.99, 1.),
            ("entropy", 1e-323, 1e-4),
            ("entropy_fall", 0.999, 0.8)
        ]

    def episode(self, env, agent, nr_episode=0):
        state = env.reset()

        undiscounted_return = 0
        done = False
        time_step = 0
        losses_dict = None
        while not done:
            if self.render_env:
                env.render()
            # 1. Select action according to policy
            action = agent.policy(state)
            # 2. Execute selected action
            next_state, reward, done, _ = env.step(action) # _ = decision_steps but are not used here
            # 3. Integrate new experience into agent
            losses_dict = agent.update(nr_episode, state, action, reward, next_state, done)
            state = next_state
            undiscounted_return += reward
            time_step += 1


        print(nr_episode, ":", undiscounted_return)
        return undiscounted_return, losses_dict

    def run_with_params(self, worker_id, run_id, params, state_dict, continue_training):
      mlflow.set_experiment(experiment_name=str(self.result_dir))
      with mlflow.start_run():
        """ Environment Setup"""
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=self.time_scale)
        unity_env = UnityEnvironment(
            file_name="Unity/worm_single_environment.x86_64" if "linux" in platform else "Unity",
            # file_name="Unity/simple_ball_environment.x86_64" if "linux" in platform else "Unity",
            worker_id=worker_id,
            no_graphics=not self.render_env,
            side_channels=[channel],
        )
        unity_env.reset()
        env = UnityToGymWrapper(unity_env)

        """ Loading old state dict """
        if state_dict:
            state_dict = torch.load(state_dict)

        if continue_training:
            print("Overriding given parameters with stored params in state_dict...")
            params = {**params, **state_dict["trainer"]["params"]}
            print("New params: ", params)

        # we only want to store configurable hyper params
        params_to_save = params.keys()
        params = {
            **params,
            "nr_input_features": env.observation_space.shape[0],  # 64
            "env": env,
            "nr_actions": 9, #env.action_space.shape[0],
            "lower_bound": env.action_space.low,
            "upper_bound": env.action_space.high,
            "type": env.action_space.dtype,
        }
        logged_params = {k:params[k] for k in params_to_save} 
        mlflow.log_params(logged_params)

        # Agent setup
        #agent = RandomAgent(params)
        agent = A2CLearner(params)
        if state_dict:
            print("Loading state dict...")
            agent.load_state_dict(state_dict["agent"])
            print("Loaded state dict.")

        results = []
        start_episode = 0
        if continue_training:
            print("Loading old training state...")
            results = state_dict["trainer"]["results"]
            start_episode = int(state_dict["trainer"]["cur_episode"]) + 1 # because this episode was finished alread
            print("Loaded old training state.")

        # train
        for i in range(start_episode, self.training_episodes):
             (reward, metrics) = self.episode(env, agent, nr_episode=i)
             results.append((reward,metrics))
             mlflow.log_metrics(metrics={"reward": reward, **metrics})
             if (i+1)% self.save_interval == 0:
                print("Saving agent state...")
                save_path = self.result_dir / "{}_episode_{:02d}.state_dict".format(run_id, i)
                torch.save({
                    "agent": agent.state_dict(),
                    "trainer":{
                        "results": results,
                        "cur_episode": i,
                        "params": {key: params[key] for key in params_to_save}
                    }
                }, save_path)
                #mlflow.log_artifact(local_path=str(save_path))
                print("Saved agent state.")

        # not needed anymore
        unity_env.close()
        returns, measurements_dicts = zip(*results)
        def flatmap(func, *iterable):
            return itertools.chain.from_iterable(map(func, *iterable))
        squeezed_measurements_dicts = defaultdict(list)
        # for key,value in [entry for d in measurements_dicts for entry in d.items()]:
        for key, value in flatmap(lambda d: d.items(), measurements_dicts):
            squeezed_measurements_dicts[key].append(value)
        return returns, squeezed_measurements_dicts