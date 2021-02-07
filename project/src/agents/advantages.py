import torch

def reinforce(Rs,):
  return Rs

def advantage_actor_critic(Rs, values):
  return Rs - values

def temporal_difference(gamma, rewards, values, next_values):
  return rewards + gamma * next_values - values

def nstep(gamma, rewards, values, next_values, n=3):
  sums = []
  for index in range(len(rewards)):
    n = min(n, len(rewards) - index -1 )
    a0 = sum([gamma**k *rewards[index + k] for k in range(n)])
    a1 = gamma ** index * next_values[index + n - 1]
    sums.append(a0 + a1)
  return torch.stack(sums) - values
  