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
        self.experiences = []
        self.batch_size = batch_size

    def store_memory(self, *elements):
        self.experiences.append(tuple(elements))

    def clear_memory(self):
        self.experiences = []

    def size(self):
        return len(self.experiences)


class ActorNet(nn.Module):
    """
    Actor NN.
    """
    
    def __init__(self, nr_input_features, nr_actions, activation, nr_hidden_units = 256):
        super().__init__()
        self.policy_base_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            PPOLearner.create_activation(activation),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            PPOLearner.create_activation(activation),
        )
        self.action_head_loc = nn.Sequential( # Actor LOC-Ausgabe
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Tanh(),
        )
        self.action_head_scale = nn.Sequential( # Actor SCALE-Ausgabe
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Softplus(),
        ) #

    def forward(self, states):
        x = self.policy_base_net(states)
        locs = self.action_head_loc(x)
        scales = self.action_head_scale(x)
        scales = scales.clamp(min=0.0, max=1)
        return locs, scales
    
class CriticNet(nn.Module):
    """
    Critic NN.
    """

    def __init__(self, nr_input_features, activation, nr_hidden_units = 256):
        super().__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            PPOLearner.create_activation(activation),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            PPOLearner.create_activation(activation),
            nn.Linear(nr_hidden_units, 1),
        )

    def forward(self, states):
        x = self.critic_net(states).squeeze(1)
        return x  
    
def parameters(actor: nn.Module,critic : nn.Module):
    {**actor.parameters(), **critic.parameters()}


DEFAULT_BATCH_SIZE = 2024
DEFAULT_BUFFER_SIZE = 20240
DEFAULT_ALPHA = 0.0003
DEFAULT_EPSILON_CLIP = 0.2
DEFAULT_LAMBDA = 0.95
DEFAULT_EPOCH = 3
DEFAULT_GAMMA = 0.995
DEFAULT_HIDDEN_NEURONS = 512

DEFAULT_ACTIVATION_FKT = "ReLu"

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
        parser.add_argument("-bs", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
        parser.add_argument("-buffs", "--buffer_size", type=int, default=DEFAULT_BUFFER_SIZE)
        parser.add_argument("-a", "--alpha", type=float, default=DEFAULT_ALPHA, help="Learning rate")
        parser.add_argument("-e", "--epsilon_clip", type=float, default=DEFAULT_EPSILON_CLIP,)
        parser.add_argument("-l", "--lambd", type=float, default=DEFAULT_LAMBDA,)
        parser.add_argument("-epoch", "--epoch", type=int, default=DEFAULT_EPOCH,)
        parser.add_argument("-g", "--gamma", type=float, default=DEFAULT_GAMMA,)
        parser.add_argument("-hn", "--hidden_neurons", type=int, default=DEFAULT_HIDDEN_NEURONS)
    
    @staticmethod
    def create_activation(activation):
      if activation == "ReLu":
        return nn.ReLU()
      elif activation == "sigmoid":
        return nn.Sigmoid()
      elif activation == "tanh":
        return nn.Tanh()
      else:
        raise Exception("Invalid activation function given.")


    @staticmethod
    def add_config_args(parser):
        parser.add_argument('-act', '--activation', type=str, default=DEFAULT_ACTIVATION_FKT, choices=["ReLu", "sigmoid", "tanh"])
        return parser

    def __init__(self,
            env, 
            batch_size=DEFAULT_BATCH_SIZE,
            buffer_size=DEFAULT_BUFFER_SIZE,
            alpha=DEFAULT_ALPHA,
            epsilon_clip=DEFAULT_EPSILON_CLIP,
            lambd=DEFAULT_LAMBDA,
            epoch=DEFAULT_EPOCH,
            gamma=DEFAULT_GAMMA,
            hidden_neurons=DEFAULT_HIDDEN_NEURONS,
            activation = DEFAULT_ACTIVATION_FKT,
            only_model=False, state_dict=None,
            **kwargs,
        ):
        super().__init__(env)
        self.config = {
            "batch_size" : batch_size,
            "buffer_size" : buffer_size,
            "alpha" : alpha,
            "epsilon_clip" : epsilon_clip,
            "lambd" : lambd,
            "epoch" : epoch,
            "gamma" : gamma,
            "hidden_neurons" : hidden_neurons,
            "activation" : activation
        }
        if state_dict:
          state_dict = state_dict["agent"]

        """On full state loading override initial params"""
        if not only_model and state_dict:
            print("Loading initial params from state dict...")
            self.config.update(state_dict["config"])
            print("Loaded initial params from state dict.")
        self.__dict__.update(self.config)

        """On full state loading override param state"""
        if not only_model and state_dict:
            print("Loaded state params from state dict.")
            self.__dict__.update(state_dict["state"])
            print("Loaded state params from state dict.")

        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        self.actor = ActorNet(
          nr_actions=self.nr_actions,
          nr_hidden_units=self.hidden_neurons,
          nr_input_features=self.nr_input_features,
          activation=self.activation,
          ).to(self.device)
        self.critic = CriticNet(
          nr_input_features=self.nr_input_features,
          nr_hidden_units=self.hidden_neurons,
          activation=self.activation,
        ).to(self.device)
        self.memory = PPOMemory(batch_size)

        if state_dict:
          print("Loading model...")
          self.actor.load_state_dict(state_dict["model-actor"])
          self.critic.load_state_dict(state_dict["model-critic"])
          print("Loaded model.")

        self.optimizer_crit = torch.optim.Adam(self.critic.parameters(), lr=self.alpha)
        self.optimizer_act = torch.optim.Adam(self.actor.parameters(), lr=self.alpha)

        """Log Params"""
        print(f"Loaded PPOLearner {str(self.config)}")
        mlflow.log_params({
            "agent": "ppo",
            **self.config,
        })

    def policy(self, state):
        """Behavioral strategy of the agent. Maps state to action."""
        states = torch.tensor([state], dtype=torch.float32, device=self.device)
        actions, _ = self.predict_policy(states)
        return actions[0].detach().clone().cpu().numpy()

    def predict_policy(self, states):
        """Behavioral strategy of the agent. Maps states to actions, log_probs."""
        locs,scales = self.actor(states)
        distributions = torch.distributions.normal.Normal(locs, scales)
        actions = distributions.sample()
        actions = actions.squeeze(1)
        actions = actions.clamp(min=-1, max=1)
        log_probs = distributions.log_prob(actions)
        return actions.detach(), log_probs

    def _compute_advantage(self, rewards, state_values, next_state_values, dones):
        list_of_tuples = list(zip(rewards, state_values, next_state_values, dones))
        advantages, A = [], 0
        for reward, state_value, next_state_value, done,  in reversed(list_of_tuples):
            if done:
                A = 0
            A *= self.gamma * self.lambd
            A += reward + self.gamma * next_state_value * (1 - int(done)) - state_value
            advantages.insert(0, A.item())
            #advantages.append(A.item())
        #advantages.reverse()
        return torch.tensor(advantages, device=self.device, dtype=torch.float32)
    

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

        self.memory.store_memory(reward, state, next_state, action, done)
        if self.memory.size() < self.buffer_size:
            return {}
        
        print("Learning from experiences")
            
        # update NNs
        rewards, states, next_states, actions, dones = zip(*self.memory.experiences)
        # convert to tensor
        #returns = self._compute_returns(rewards)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

        #recompute scales and locs for measures
        action_locs, action_scales = self.actor(states)
        action_locs = action_locs.detach()
        action_scales = action_scales.detach()

        # Remove old policy values from gradient graph
        state_values = self.critic(states).detach() 
        next_state_values = self.critic(next_states).detach() 
        _, old_log_probs = self.predict_policy(states) 
        old_log_probs = old_log_probs.detach() #ditto

        # compute advantages
        advantages = self._compute_advantage(rewards, state_values, next_state_values, dones) # dones

        actor_loss_list, critic_loss_list, loss_list = [],[],[]

        for _ in range(self.epoch):
            for b_states, b_state_values, b_actions, b_old_log_probs, b_advantages \
                in self.generate_batches(states, state_values, actions, old_log_probs, advantages):

                _ , b_new_log_probs = self.predict_policy(b_states)
                b_new_state_values = self.critic(b_states)

                # batch actor loss
                prob_ratio = b_new_log_probs.exp() / b_old_log_probs.exp()

                b_advantages_expanded =b_advantages.unsqueeze(1) #.expand(-1, self.nr_actions)
                prob_ratio_weighted = prob_ratio * b_advantages_expanded
                prob_ratio_weighted_clipped = prob_ratio.clamp(min=1-self.epsilon_clip, max=1+self.epsilon_clip) * b_advantages_expanded

                b_actor_loss = -torch.min(prob_ratio_weighted, prob_ratio_weighted_clipped).mean()

                #b_advantages = breturns - b_state_values
                # batch critic loss, actor critic advantage
                b_returns = b_advantages + b_state_values 
                b_critic_loss = F.mse_loss(target=b_returns, input=b_new_state_values)
                b_loss = b_actor_loss + b_critic_loss

                # store losses in lists for measurements
                critic_loss_list.append(b_critic_loss.item())
                actor_loss_list.append(b_actor_loss.item())
                loss_list.append(b_loss.item())
                
                self.optimizer_act.zero_grad()
                self.optimizer_crit.zero_grad()
                b_loss.backward()
                self.optimizer_act.step()
                self.optimizer_crit.step()

        measures = {
            "loss": np.array(loss_list).mean(),
            "actor_loss": np.array(actor_loss_list).mean(),
            "critic_loss" : np.array(critic_loss_list).mean(),
            "advantages_std" : advantages.std().item(),
            "action_loc_std": action_locs.std().item(),
            "action_scale_avg": action_scales.mean().item(),
            "action_scale_std": action_scales.std().item(),
            "state_value_avg": state_values.mean().item(),
            "state_value_std": state_values.std().item(),
        }
        mlflow.log_metrics(measures, step=nr_episode)

        self.memory.clear_memory()
        return measures

    def state_dict(self):
        return {
          "model-critic": self.critic.state_dict(),
          "model-actor": self.actor.state_dict(),
          "config": self.config,
          "state": {k: self.__dict__.get(k) for k in self.config},
        }
