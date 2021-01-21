import random
import math
import mlflow
import numpy as np
from numpy.core.fromnumeric import shape, std
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import mlflow

from .agent import Agent

class PPOMemory:
    """
    Batch Container.
    """

    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNet(nn.Module):
    """
    Actor NN.
    """
    
    def __init__(self, nr_input_features, nr_actions, nr_hidden_units = 256):
        super().__init__()
        self.policy_base_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        self.action_head_loc = nn.Sequential( # Actor LOC-Ausgabe
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Tanh(),
        )
        self.action_head_scale = nn.Sequential( # Actor SCALE-Ausgabe
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Softplus(),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003) # 0.0003
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        self.to(self.device)

    def forward(self, states):
        x = self.policy_base_net(states)
        locs = self.action_head_loc(x)
        scales = self.action_head_scale(x)
        scales.clamp(min=0.01, max=1) # min clamping could be excluded
        return torch.distributions.normal.Normal(locs, scales)
    
class CriticNet(nn.Module):
    """
    Critic NN.
    """

    def __init__(self, nr_input_features, nr_hidden_units = 256):
        super().__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        self.to(self.device)

    def forward(self, states):
        x = self.critic_net(states)
        return x  
    
DEFAULT_BATCH_SIZE = 2024
DEFAULT_BUFFER_SIZE = 20240
DEFAULT_ALPHA = 0.0003
DEFAULT_BETA = 0.005
DEFAULT_EPSILON_CLIP = 0.02
DEFAULT_LAMBDA = 0.95
DEFAULT_EPOCH = 3
DEFAULT_GAMMA = 0.995
DEFAULT_HIDDEN_NEURONS = 512

class PPOLearner(Agent):
    """
    Base class of an autonomously acting and learning agent.
    """

    """ Returns list of tuples (name, min, max)"""
    @staticmethod
    def hyper_params():
        return {
            "epsilon_clip": {"min": 0.2, "max": 0.2},
            "name": {"min": 0, "max": 1},
            "lambd": {"min": 0.95, "max": 0.95},
        }

    @staticmethod
    def agent_name():
        return "ppo"
    
    @staticmethod
    def add_hyper_param_args(parser: ArgumentParser):
        parser.add_argument("-bs", "--batch_size", type=int, default=2024)
        parser.add_argument("-buffs", "--buffer_size", type=int, default=20240)
        parser.add_argument("-a", "--alpha", type=float, default=0.0003, help="Learning rate")
        parser.add_argument("-b", "--beta", type=float, default=0.005,)
        parser.add_argument("-e", "--epsilon_clip", type=float, default=0.02,)
        parser.add_argument("-l", "--lambd", type=float, default=0.95,)
        parser.add_argument("-epoch", "--epoch", type=int, default=3,)
        parser.add_argument("-g", "--gamma", type=float, default=0.995,)
        parser.add_argument("-hn", "--hidden_neurons", type=int, default=512)
    
    @staticmethod
    def add_config_args(parser):
        return parser

    def __init__(self,
            env, 
            batch_size=DEFAULT_BATCH_SIZE,
            buffer_size=DEFAULT_BUFFER_SIZE,
            alpha=DEFAULT_ALPHA,
            beta=DEFAULT_BETA,
            epsilon_clip=DEFAULT_EPSILON_CLIP,
            lambd=DEFAULT_LAMBDA,
            epoch=DEFAULT_EPOCH,
            gamma=DEFAULT_GAMMA,
            hidden_neurons=DEFAULT_HIDDEN_NEURONS,
            **kwargs,
        ):
        super().__init__(env)
        params = {
            "batch_size" : batch_size,
            "buffer_size" : buffer_size,
            "alpha" : alpha,
            "beta" : beta,
            "epsilon_clip" : epsilon_clip,
            "lambd" : lambd,
            "epoch" : epoch,
            "gamma" : gamma,
            "hidden_neurons" : hidden_neurons,
        }
        self.__dict__.update(params)
        mlflow.log_params(params)

        self.actor = ActorNet(self.nr_input_features, self.nr_actions)
        self.critic = CriticNet(self.nr_input_features)
        self.memory = PPOMemory(batch_size)

    def policy(self, state):
        """Behavioral strategy of the agent. Maps state to action."""
        states = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        actions, _ = self.predict_policy(states)
        return actions[0].detach().clone().numpy()

    def predict_policy(self, states):
        """Behavioral strategy of the agent. Maps states to actions, log_probs."""
        distributions = self.actor(states)
        actions = distributions.sample().squeeze(1) # this could be a bug
        actions = actions.detach()
        log_probs = distributions.log_prob(actions)
        return actions, log_probs

    def _compute_advantage(self, rewards, state_values, next_state_values):
        #next_state_values = [*state_values[1::], 0.]# may fail :)
        list_of_tuples = list(zip(rewards, state_values, next_state_values))
        advantages, A = [], 0
        for reward, state_value, next_state_value, in reversed(list_of_tuples):  #this could be a bug (bsc 2021) - last next_state_value has to be 0
            A = reward + self.gamma * next_state_value - state_value + self.gamma * self.lambd * A
            advantages.append(A)
        advantages.reverse()
        return advantages

    def generate_batches(self, *tensors):
        memory_len = len(tensors[0])
        batch_starts = np.arange(0, memory_len, self.batch_size, dtype=np.int64)
        np.random.shuffle(batch_starts)
        # for each batch, create slice for each given tensor and store them as tuple of tensor slices in a list
        batches = [tuple((t[i:i + self.batch_size] for t in tensors)) for i in batch_starts]
        return batches

    def update(self,nr_episode, state, action, reward, next_state, done):
        """
        Learning method of the agent. Integrates experience into
        the agent's current knowledge.
        """

        # store experience
        if not done:
            vals = self.critic(torch.from_numpy(state))
            probs = self.actor(torch.from_numpy(state))
            self.memory.store_memory(state, action, probs, vals, reward, done) 
            # vals is only one value vector = tensor([-0.0590], grad_fn=<AddBackward0>)
            # probs = Normal(loc: torch.Size([2]), scale: torch.Size([2]))
            return {}
      
        for _ in range(self.epoch):

            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1].item()*\
                            (1-int(dones_arr[k])) - values[k].item())
                    discount *= self.gamma*self.lambd
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.actor.device)
            # values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                #old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                #total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()


        self.memory.clear_memory()    
        return {}

        '''
            # update NNs
            #rewards, states, next_states, actions = zip(*self.memory.experiences)

            # convert to tensor
            rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
            states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
            next_states = torch.tensor(next_states, dtype=torch.float).to(self.actor.device)
            actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)

            # Remove old policy values from gradient graph
            state_values = self.critic(states).detach() 
            next_state_values = self.critic(next_states).detach() 
            _, old_log_probs = self.predict_policy(states) 
            old_log_probs = old_log_probs.detach() #ditto

            # compute advantages
            advantages = torch.stack(self._compute_advantage(rewards, state_values, next_state_values))

            for b_states, b_state_values, b_actions, b_old_log_probs, b_advantages \
                in self.generate_batches(states, state_values, actions, old_log_probs, advantages):

                _ , b_new_log_probs = self.predict_policy(b_states)
                b_new_state_values = self.critic(b_states)

                # batch actor loss
                prob_ratio = b_new_log_probs.exp() / b_old_log_probs.exp()
                prob_ratio_weighted = prob_ratio *b_advantages
                prob_ratio_weighted_clipped = prob_ratio.clamp(min=1-self.epsilon_clip, max=self.epsilon_clip) * b_advantages
                b_actor_loss = -torch.min(prob_ratio_weighted, prob_ratio_weighted_clipped).mean()

                # batch critic loss
                b_returns = b_advantages + b_state_values
                b_critic_loss = F.mse_loss(b_returns, b_new_state_values)

                b_loss = b_actor_loss + 0.5*b_critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                b_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        return {}
        '''

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, only_model, strict=False):
        pass
