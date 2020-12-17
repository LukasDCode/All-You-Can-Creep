import rooms
import a2c as a
import matplotlib.pyplot as plot
import gym

def episode(env, agent, nr_episode=0):
    state = env.reset()
    undiscounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
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
    
params = {}
# Domain setup
env = gym.make('MountainCar-v0')
env = gym.make('Acrobot-v1')
env = gym.make('CartPole-v1')
params["nr_actions"] = env.action_space.n
params["nr_input_features"] = env.observation_space.shape[0]
params["env"] = env

# Hyperparameters
params["gamma"] = 0.99
params["alpha"] = 0.001
training_episodes = 2000

# Agent setup
agent = a.A2CLearner(params)
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.show()
