import argparse
import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
from sys import platform
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def create_env():
    """Environment Setup"""
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=20)
    unity_env = UnityEnvironment(
        file_name="../Unity/simple_ball_environment.x86_64" if "linux" in platform else "../../WormDynamic", #WormDynamic #Ball
        worker_id=0,
        no_graphics=True,
        side_channels=[channel]
    )
    unity_env.reset()
    env = UnityToGymWrapper(
        unity_env
        )
    return env


def run(args):

    env = create_env()
    env.reset()

    nr_input_features = env.observation_space.shape[-1]
    nr_actions = env.action_space.shape[0]
    lower_bound = env.action_space.low[0]
    upper_bound = env.action_space.high[0]

    print()
    print("Environment loaded!")
    print()
    print("  nr_input_features :", env.observation_space.shape[-1])
    print("  nr_actions        :", env.action_space.shape[0])
    print("  lower_bound       :", env.action_space.low[0])
    print("  upper_bound       :", env.action_space.high[0])

    N = args["buffer_size"]
    batch_size = args["batch_size"]
    n_epochs = args["n_epochs"]
    alpha1 = args["alpha1"]
    alpha2 = args["alpha2"]
    n_games = args["episodes"]

    print()
    print("Welcome to our Training 8-)")
    print()
    print("  We will train for",str(n_games)," episodes!")
    print("  Learning rate actor NN  :", str(alpha1))
    print("  Learning rate critic NN :", str(alpha2))
    print("  Batch size              :", str(batch_size))
    print("  Buffer size             :", str(N))
    print("  Epochs of updates       :", str(n_epochs))
    print()
    print()

    agent = Agent(n_actions=nr_actions, batch_size=batch_size, 
                    alpha1=alpha1,alpha2=alpha2, n_epochs=n_epochs, 
                    input_dims=nr_input_features)

    figure_file = 'plots/ball/' + str(n_games) + 'a1' + str(alpha1) + 'a2' + str(alpha2) + 'bs' + str(batch_size) + 'N' + str(N) + 'n_epochs' + str(n_epochs) + '.png'

    agent.load_models()

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        #print("start env.reset")
        observation = env.reset()
        #print("end env.reset")
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                print("start agent.learn")
                agent.learn()
                print("end agent.learn")
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
        print("Saving Model at Step:", i, "with average:", avg_score)
        agent.save_models()

        if i%1 == 0:
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,\
                'time_steps', n_steps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run worms with hyper params')

    parser.add_argument('-n','--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('-e','--n_epochs', type=int, default=4, help='Epochs of updates')
    parser.add_argument('-bas', '--batch_size', type=int, default=2024, help='Batch size')
    parser.add_argument('-a1', '--alpha1', type=float, default=0.0003, help='Learning rate actor NN')
    parser.add_argument('-a2', '--alpha2', type=float, default=0.0003, help='Learning rate critic NN')
    parser.add_argument('-bus', '--buffer_size', type=int, default=20240, help='Buffer size, should be multiple batch size')

    run(vars(parser.parse_args()))

