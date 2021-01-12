from sys import platform
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np

DEFAULT_TIME_SCALE = 20
DEFAULT_VISUALIZE = False

class WormDomain():

    @staticmethod
    def add_parse_args(parser):
      parser.add_argument('-v', '--visualize', action='store_true', default=DEFAULT_VISUALIZE, help='call env.render')
      parser.add_argument('-t', '--time_scale', type=float, default=DEFAULT_TIME_SCALE, help='simulation speed scaling, higher values make physics less precise')

    def __init__(self, visualize=False, time_scale=20, **kwargs):
        self.__dict__.update({
            'visualize': visualize,
            'time_scale': time_scale,
        })

    def evaluate(self, states):
        distances = np.array(states)[:,4]
        return {
            "min_distance": float(min(distances)),
            "max_distance": float(max(distances)),
            "avg_distance": float(distances.mean()),
            "last_distance": float(distances[-1]),
        }

    def create_env(self, worker_id, **kwargs):
        """ Environment Setup"""
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=self.time_scale)
        unity_env = UnityEnvironment(
            #file_name="Unity/worm_single_environment.x86_64" if "linux" in platform else "Unity",
            file_name="Unity/simple_ball_environment.x86_64" if "linux" in platform else "Unity",
            worker_id=worker_id,
            no_graphics=not self.visualize,
            side_channels=[channel],
        )
        unity_env.reset()
        env = UnityToGymWrapper(unity_env)
        return env
