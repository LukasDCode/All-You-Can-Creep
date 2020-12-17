import pdb
from sys import platform
import rooms
import a2c_worm as a2c
import matplotlib.pyplot as plot
import gym

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def episode(env, agent, nr_episode=0):
    state = env.reset()
    undiscounted_return = 0
    discount_factor = 0.99
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


params = {}
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

"""
print(env.action_space) # --> Box(-1.0, 1.0, (9,), float32)
print(env.action_space.shape[0]) # --> 9
print(env.observation_space) # --> Box(-inf, inf, (64,), float32)
print(env.observation_space.shape[0]) # --> 64
"""

params["nr_input_features"] = env.observation_space.shape[0] # 64
params["env"] = env

params["nr_actions"] = env.action_space.shape[0] # 9
params["lower_bound"] = env.action_space.low
params["upper_bound"] = env.action_space.high
params["type"] = env.action_space.dtype

"""
print("size")
print(params["nr_actions"])
print("Lower Bound")
print(params["lower_bound"])
print("Upper Bound")
print(params["upper_bound"])
print("Type")
print(params["type"])
"""

# Hyperparameters
params["gamma"] = 0.99
params["alpha"] = 0.001
training_episodes = 2000

# Agent setup
agent = a2c.A2CLearner(params)
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.show()

