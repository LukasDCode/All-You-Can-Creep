import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve
from sys import platform
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


def create_env():
    """ Environment Setup"""
    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=2)
    unity_env = UnityEnvironment(
        file_name="../Unity/ball.x86_64" if "linux" in platform else "Unity",
        worker_id=0,
        no_graphics=False,
        side_channels=[channel]
    )
    unity_env.reset()
    env = UnityToGymWrapper(
        unity_env
        )
    return env

if __name__ == '__main__':
    env = create_env()
    env.reset()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=2, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=8)
    #agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
    #                alpha=alpha, n_epochs=n_epochs, 
    #                input_dims=env.observation_space.shape)
    n_games = 1000 # 300

    figure_file = 'plots/ball_environment.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


