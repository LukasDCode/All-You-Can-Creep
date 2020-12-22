import random
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .agent import Agent

class A2CNet(nn.Module):
    def __init__(self, nr_input_features, nr_actions):
        super(A2CNet, self).__init__()
        nr_hidden_units = 64
        self.fc_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        self.action_head_loc = nn.Sequential( # Actor LOC-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            #nn.Tanh()
        ) 
        self.action_head_scale = nn.Sequential( # Actor SCALE-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            #nn.Softplus()  
        ) # Actor = Policy-Function NN
        self.value_head = nn.Linear(nr_hidden_units, 1) # Critic = Value-Function NN

    def forward(self, x):
        x = self.fc_net(x)
        # x = x.view(x.size(0), -1) # reshapes the tensor
        return (self.action_head_loc(x), self.action_head_scale(x)) ,  self.value_head(x)


"""
 Autonomous agent using Synchronous Actor-Critic.
"""
class A2CLearner(Agent):

    def __init__(self, params):
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.alpha = params["alpha"]
        self.nr_input_features = params["nr_input_features"]
        self.transitions = []
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.a2c_net = A2CNet(self.nr_input_features, self.nr_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.a2c_net.parameters(), lr=params["alpha"])

    """
     Samples a new action using the policy network.
    """
    def policy(self, state):
        (action_locs, action_scales), _ = self.predict_policy([state])
        # print("Actions {} {}".format( action_locs, action_scales))
        m = torch.distributions.normal.Normal(action_locs, action_scales)
        return m.sample().detach().cpu().numpy()


    """
     Predicts the action probabilities.
    """       
    def predict_policy(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        return self.a2c_net(states)
        
    """
     Predicts the state values.
    """       
    def predict_value(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        return self.value_net(states)

        
    """
     Performs a learning update of the currently learned policy and value function.
    """
    def update(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        # 64 Werte fuer state, 9 Werte fuer action, 1 Wert fuer reward, 64 Werte fuer next_state, 1 boolean fuer done

        if done:
            states, actions, rewards, next_states, dones = tuple(zip(*self.transitions))
            
            # Calculate and normalize discounted returns
            discounted_returns = []
            R = 0
            for reward in reversed(rewards):
                R = reward + self.gamma*R
                discounted_returns.append(R)
            discounted_returns.reverse()
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
            discounted_returns = torch.tensor(discounted_returns, device=self.device, dtype=torch.float).detach()
            normalized_returns = (discounted_returns - discounted_returns.mean())
            normalized_returns /= (discounted_returns.std() + self.eps)

            # Calculate losses of policy and value function
            actions = torch.tensor(actions, device=self.device, dtype=torch.float)
            (action_locs, action_scales), state_values = self.predict_policy(states) # Tupel + value_head --- return aus Zeile 29: tupel((action_probs_loc, action_probs_scale), state_values)
            states = torch.tensor(states, device=self.device, dtype=torch.float)


            def _loss_basic():
                loc_losses, scale_losses = [], []
                for action_loc, action_scale, action, value, R in zip(action_locs, action_scales, actions, state_values, normalized_returns):
                    advantage = R - value.item()
                    loc_loss = F.mse_loss(input=action_loc, target=action) * advantage
                    scale_loss = (nn.Softplus()(action_scale) * advantage).mean()

                    loc_losses.append(loc_loss)
                    scale_losses.append(scale_loss)

                print("loc_losses")
                print(loc_losses[-1])

                print("scale_losses")
                print(scale_losses[-1])

                return torch.stack(loc_losses).sum() + torch.stack(scale_losses).sum()

            def _loss_other():
                policy_losses_loc = []
                policy_losses_scale = []
                value_losses = []
                for action_loc, action_scale, action, value, R in zip(action_locs, action_scales, actions, state_values, normalized_returns):
                  ENTROPY_BETA = 1e-4 # vielleicht als Hyperparameter
                  advantage = R - value.item()
                  loss_value = F.mse_loss(value.squeeze(-1), R)
                  def calc_logprob():
                      p1 = - ((action_loc - action) ** 2) / (2*action_scale.clamp(min=1e-3))
                      p2 = - torch.log(torch.sqrt(2 * math.pi * action_scale))
                      return p1 + p2
                  log_prob = advantage * calc_logprob()
                  loss_policy = -log_prob.mean()
                  entropy_loss = ENTROPY_BETA * (-(torch.log(2*math.pi*action_scale) + 1)/2).mean()
  
                  policy_losses_loc.append(loss_policy)
                  policy_losses_scale.append(entropy_loss)
                  value_losses.append(loss_value)
                
                return torch.stack(policy_losses_loc).sum() + torch.stack(policy_losses_scale).sum() + torch.stack(value_losses).sum()
            
            loss = _loss_basic()

            # Optimize joint loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Don't forget to delete all experiences afterwards! This is an on-policy algorithm.
            self.transitions.clear()

            return loss

        return None
