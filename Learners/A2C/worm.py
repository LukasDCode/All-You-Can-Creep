import pdb
from sys import platform
import rooms
import a2c_worm as a2c
import matplotlib.pyplot as plot
import gym

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from ...Tuning.executor import Domain as DomainTrainingAdaptor


def episode(env, agent, nr_episode=0):
    state = env.reset()
    undiscounted_return = 0
    done = False
    time_step = 0
    while not done:
        # env.render()
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


def run_with_params(training_episodes,params,):
  params = params.copy()

  # Domain setup
  # Environment
  channel = EngineConfigurationChannel()
  channel.set_configuration_parameters(time_scale = 1.0)

  if platform == "linux" or platform == "linux2":
    # linux
    unity_env = UnityEnvironment(file_name="../../Unity/worm_single_environment.x86_64", no_graphics=False, side_channels=[channel])
  elif platform == "win32":
    # Windows...
    unity_env = UnityEnvironment(file_name="..\\..\\Unity", no_graphics=False, side_channels=[channel])
  env = UnityToGymWrapper(unity_env)

  params["nr_input_features"] = env.observation_space.shape[0] # 64
  params["env"] = env
  params["nr_actions"] = env.action_space.shape[0] # 9
  params["lower_bound"] = env.action_space.low
  params["upper_bound"] = env.action_space.high
  params["type"] = env.action_space.dtype

  # Agent setup
  agent = a2c.A2CLearner(params)
  # train
  returns = [episode(env, agent, i) for i in range(training_episodes)]
  return returns

class WormDomainAdaptor(DomainTrainingAdaptor):

    def run(self,params):
        training_episodes = 2000
        return run_with_params(training_episodes, params)

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

def main():
  params = {}
  # Hyperparameters
  params["gamma"] = 0.99
  params["alpha"] = 0.001
  training_episodes = 2000
  returns = run_with_params(training_episodes,params)

  x = range(training_episodes)
  y = returns
  plot.plot(x, y)
  plot.title("Progress")
  plot.xlabel("episode")
  plot.ylabel("undiscounted return")
  plot.show()

if __name__ == "__main__":
  main()
