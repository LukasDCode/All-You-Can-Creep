
from os import close, stat
import pdb
from sys import platform
import matplotlib.pyplot as plot
import gym
import argparse
import json
from uuid import uuid4
from collections import defaultdict
from pathlib import Path
import itertools

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from .agents.a2c import A2CLearner
from ..exec.executor import Domain as DomainTrainingAdaptor, Executor 


class WormDomainAdaptor(DomainTrainingAdaptor):

    @staticmethod
    def add_parse_args(parser):
      parser.add_argument('-n','--episodes', type=int, default=2000, help='training episodes')
      parser.add_argument('-v', '--visualize', type=bool, default=False, help='call env.render')  
      parser.add_argument('-t', '--time_scale', type=float, default=20, help='simulation speed scaling, higher values make physics less precise')
      parser.add_argument('-r', '--result', type=str, default="result", help='file base name to save results into')
      return parser


    def __init__(self, config):
        super().__init__()
        self.render_env = config.visualize
        self.training_episodes = config.episodes
        self.result_base_name = config.result
        self.scale = config.time_scale

    def run(self, worker_id, params):
        (rewards, losses_dicts) = self.run_with_params(
            worker_id=worker_id,
            params=params,
        )
        result_dump = {
            "algorithm": "a2c",
            "params": params,
            "measures": {"rewards": rewards, **losses_dicts}
        }
        result_path = Path(self.result_base_name + str(uuid4()) + ".json")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open('w+') as file:
            json.dump(result_dump, file, indent=2)
        return rewards

    def param_dict(self):
        return [
            ("alpha", 0.001, 0.001),
            ("gamma", 0.99, 1.),
            ("entropy", 1e-323, 1e-4)
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
            next_state, reward, done, _ = env.step(action)
            # 3. Integrate new experience into agent
            losses_dict = agent.update(state, action, reward, next_state, done)
            state = next_state
            undiscounted_return += reward
            time_step += 1
        print(nr_episode, ":", undiscounted_return)
        return undiscounted_return, losses_dict

    def run_with_params(self, worker_id, params,):
        params = params.copy()

        # Domain setup
        # Environment
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=self.scale)

        unity_env = UnityEnvironment(
            file_name="Unity/worm_single_environment.x86_64" if "linux" in platform else "Unity",
            # file_name="Unity/simple_ball_environment.x86_64" if "linux" in platform else "Unity",
            worker_id=worker_id,
            no_graphics=not self.render_env,
            side_channels=[channel],
        )

        env = UnityToGymWrapper(unity_env)
        params["nr_input_features"] = env.observation_space.shape[0]  # 64
        params["env"] = env
        params["nr_actions"] = env.action_space.shape[0]  # 9
        params["lower_bound"] = env.action_space.low
        params["upper_bound"] = env.action_space.high
        params["type"] = env.action_space.dtype

        # Agent setup
        agent = A2CLearner(params)
        # train
        results = [
            self.episode(env, agent, nr_episode=i,)
            for i in range(self.training_episodes)]

        # not needed anymore
        unity_env.close()

        returns, losses_dicts = zip(*results)

        def flatmap(func, *iterable):
            return itertools.chain.from_iterable(map(func, *iterable))
        squeezed_losses_dicts = defaultdict(list)
        # for key,value in [entry for d in losses_dicts for entry in d.items()]:
        for key, value in flatmap(lambda d: d.items(), losses_dicts):
            squeezed_losses_dicts[key].append(value)
        return returns, squeezed_losses_dicts