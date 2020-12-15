import rooms
import dqn as a
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
params["epsilon"] = 0.1
params["memory_capacity"] = 5000
params["warmup_phase"] = 2500
params["target_update_interval"] = 1000
params["minibatch_size"] = 32
params["epsilon_linear_decay"] = 1.0/params["memory_capacity"]
params["epsilon_min"] = 0.01
training_episodes = 1000

# Agent setup
agent = a.DQNLearner(params)
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.show()
