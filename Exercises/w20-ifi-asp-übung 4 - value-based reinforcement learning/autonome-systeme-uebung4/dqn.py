import random
import numpy
import copy
import math
from multi_armed_bandits import epsilon_greedy
from agent import QLearner
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):
    def __init__(self, nr_input_features, nr_actions):
        super(DQNNet, self).__init__()
        nr_hidden_units = 32
        self.fc_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        self.value_head = nn.Linear(nr_hidden_units, nr_actions)

    def forward(self, x):
        x = self.fc_net(x)
        x = x.view(x.size(0), -1)
        return self.value_head(x)

"""
 Experience Buffer for Deep RL Algorithms.
"""
class ReplayMemory:

    def __init__(self, size):
        self.transitions = []
        self.size = size

    def save(self, transition):
        self.transitions.append(transition)
        if len(self.transitions) > self.size:
            self.transitions.pop(0)

    def sample_batch(self, minibatch_size):
        nr_episodes = len(self.transitions)
        if nr_episodes > minibatch_size:
            return random.sample(self.transitions, minibatch_size)
        return self.transitions

    def clear(self):
        self.transitions.clear()

    def size(self):
        return len(self.transitions)

"""
 Autonomous agent using Deep Q-Learning.
"""
class DQNLearner(QLearner):

    def __init__(self, params):
        super(DQNLearner, self).__init__(params)
        self.device = torch.device("cpu")
        self.nr_input_features = params["nr_input_features"]
        self.epsilon = 1
        self.epsilon_linear_decay = params["epsilon_linear_decay"]
        self.epsilon_min = params["epsilon_min"]
        self.warmup_phase = params["warmup_phase"]
        self.minibatch_size = params["minibatch_size"]
        self.memory = ReplayMemory(params["memory_capacity"])
        self.training_count = 0
        self.target_update_interval = params["target_update_interval"]
        self.policy_net = DQNNet(self.nr_input_features, self.nr_actions)
        self.target_net = DQNNet(self.nr_input_features, self.nr_actions)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.update_target_network()

    """
     Overwrites target network weights with currently trained weights.
    """
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    """
     Selects a new action using epsilon-greedy exploration w.r.t. currently learned Q-Values.
    """
    def policy(self, state):
        Q_values = self.Q([state]).detach().numpy()[0]
        return epsilon_greedy(Q_values, None, epsilon=self.epsilon)
        
    """
     Predicts the currently learned Q-Values for a given batch of states.
    """
    def Q(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        return self.policy_net(states)

    """
     Predicts the previously learned Q-Values for a given batch of states.
    """
    def target_Q(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        return self.target_net(states)
        
    """
     Performs a learning update of the currently learned value function approximation.
    """
    def update(self, state, action, reward, next_state, done):
        self.memory.save((state, action, reward, next_state, done))
        self.warmup_phase = max(0, self.warmup_phase - 1)
        loss = None
        if self.warmup_phase == 0:
            self.training_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_linear_decay)
            minibatch = self.memory.sample_batch(self.minibatch_size)
            states, actions, rewards, next_states, dones = tuple(zip(*minibatch))
            non_final_mask = torch.tensor([not done for done in dones], device=self.device, dtype=torch.bool)
            next_states = [s for s, done in zip(next_states, dones) if not done]
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
            current_Q_values = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_Q_values = torch.zeros(self.minibatch_size, device=self.device)
            next_Q_values[non_final_mask] = self.target_Q(next_states).max(1)[0].detach()
            Q_targets = rewards + self.gamma*next_Q_values
            self.optimizer.zero_grad()
            loss = F.mse_loss(current_Q_values, Q_targets)
            loss.backward()
            self.optimizer.step()
            if self.training_count%self.target_update_interval == 0:
                self.update_target_network()
        return loss
    