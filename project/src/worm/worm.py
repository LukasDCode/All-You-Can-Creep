import pdb
from sys import platform
import matplotlib.pyplot as plot
import gym
import argparse
import json
from uuid import uuid4

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from .a2c_worm import A2CLearner
from ..tuning.executor import Domain as DomainTrainingAdaptor
from ..tuning.executor import Executor


def episode(env, agent, nr_episode=0, env_render=False):
    state = env.reset()

    undiscounted_return = 0
    done = False
    time_step = 0
    while not done:
        if env_render:
          env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        undiscounted_return += reward
        time_step += 1
    print(nr_episode, ":", undiscounted_return)
    return undiscounted_return


def run_with_params(env_render, training_episodes,params,):
  params = params.copy()

  # Domain setup
  # Environment
  channel = EngineConfigurationChannel()
  channel.set_configuration_parameters(time_scale = 100.0)

  if platform == "linux" or platform == "linux2":
    # linux
    unity_env = UnityEnvironment(file_name="Unity/worm_single_environment.x86_64", no_graphics=False, side_channels=[channel])
  elif platform == "win32":
    # Windows...
    unity_env = UnityEnvironment(file_name="Unity", no_graphics=False, side_channels=[channel])
  env = UnityToGymWrapper(unity_env)

  params["nr_input_features"] = env.observation_space.shape[0] # 64
  params["env"] = env
  params["nr_actions"] = env.action_space.shape[0] # 9
  params["lower_bound"] = env.action_space.low
  params["upper_bound"] = env.action_space.high
  params["type"] = env.action_space.dtype

  # Agent setup
  agent = A2CLearner(params)
  # train
  returns = [episode(
    env, 
    agent, 
    nr_episode=i, 
    env_render=env_render) for i in range(training_episodes)]
  return returns

class WormDomainAdaptor(DomainTrainingAdaptor):

    def __init__(self, config):
        super().__init__()
        self.render_env = config.visualize
        self.training_episodes = config.episodes
        self.result_base_name = config.result

    def run(self,params):
        (rewards) = run_with_params(
          env_render=self.render_env, 
          training_episodes=self.training_episodes, 
          params=params)
        result_dump = {
          "algorithm": "a2c",
          "params" : params,
          "rewards" : rewards,
          "exploration": [],
        }
        with open(self.result_base_name + str(uuid4()) +".json") as file:
          json.dump(result_dump, file)
        return rewards
    
    def param_dict(self,):
        pass

    def python_run_command(self,params):
        """Specifies the command to be run by slurm within the repository"""
        # FIXME proper solution
        # argparse interface implementation 
        # create command passing relevant information
        pass

    def python_run_parse_log(self, logfilestring):
        """Parses the slurm log to yield the requested values
        returns rewards [float]
        """
        # FIXME proper solution
        # filter with episodes and collect values
        pass


def parse_config():
  parser = argparse.ArgumentParser(description='Run worms with hyper params')
  parser.add_argument('-a','--alpha', type=float, default=0.001, help='the learning rate')
  parser.add_argument('-g','--gamma', type=float, default=0.99 ,help='the discount factor for rewards')
  parser.add_argument('-n','--episodes', type=int, default=2000, help='training episodes')
  parser.add_argument('-v', '--visualize', type=bool, default=False, help='call env.render')
  parser.add_argument('-r', '--result', type=str, default="result", help='file base name to save results into')
  return parser.parse_args()


def main():
  config = parse_config()
  print("Run with {}", str(config))
  domain = WormDomainAdaptor(config)
  with Executor(tasks_in_parallel=1, on_slurm=False, domain=domain) as executor:
    # Hyperparameters
    params = {}
    params["gamma"] = config.gamma
    params["alpha"] = config.alpha
    future = executor.submit_task(params)
    print(future.get())

if __name__ == "__main__":
  main()
