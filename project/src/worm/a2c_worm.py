import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
        self.action_head_loc = nn.Linear(nr_hidden_units, nr_actions) # Actor = Policy-Function NN
        self.action_head_scale = nn.Linear(nr_hidden_units, nr_actions) # Actor = Policy-Function NN
        # hidden amount of input nodes, 9 output nodes for 9 different actions
        self.value_head = nn.Linear(nr_hidden_units, 1) # Critic = Value-Function NN
        # hidden amount of input nodes, 1 output node for single Q value

    def forward(self, x):
        x = self.fc_net(x)
        # x = x.view(x.size(0), -1) # reshapes the tensor
        return (self.action_head_loc(x), self.action_head_scale(x)) ,  self.value_head(x)


"""
 Autonomous agent using Synchronous Actor-Critic.
"""
class A2CLearner:

    def __init__(self, params):
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.alpha = params["alpha"]
        self.nr_input_features = params["nr_input_features"]
        self.transitions = []
        self.device = torch.device("cpu")
        self.a2c_net = A2CNet(self.nr_input_features, self.nr_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.a2c_net.parameters(), lr=params["alpha"])

    """
     Samples a new action using the policy network.
    """
    def policy(self, state):
        (action_locs, action_scales), _ = self.predict_policy([state])
        # print("Actions {} {}".format( action_locs, action_scales))
        m = torch.distributions.normal.Normal(action_locs, action_scales)
        return m.sample().detach().numpy()

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
        # print("letzte transition")
        # print(self.transitions[-1])
        loss = None
        if done:
            states, actions, rewards, next_states, dones = tuple(zip(*self.transitions))
            discounted_returns = []
            
            # Calculate and normalize discounted returns
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
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)
            action_probs, state_values = self.predict_policy(states) # Tupel + value_head --- return aus Zeile 29: tupel((action_probs_loc, action_probs_scale), state_values)
            states = torch.tensor(states, device=self.device, dtype=torch.float)
            policy_losses = []

            policy_losses_loc = []
            policy_losses_scale = []

            value_losses = []
            for probs_loc, probs_scale, action, value, R in zip(action_probs[0], action_probs[1], actions, state_values, normalized_returns):
            #for probs, action_loc,action_scale, value, R in zip(action_probs, actions[0], actions[1], state_values, normalized_returns):
                advantage = R - value.item()
                
                # Zeile drunter ist nur wichtig für den diskreten Fall
                # m = torch.distributions.normal.Normal(probs_loc, probs_scale)
                
                # in der Zeile drunter steht der 'alte' policy_loss
                # policy_losses.append(-m.log_prob(action) * advantage) # tensor([[ nan, -0.3500, nan, -0.7136, -0.0231, nan, nan, nan, 0.0866]], grad_fn=<MulBackward0>)
                # policy_losses.append((F.mse_loss(action, probs_loc) + F.softplus(probs_scale))*advantage) # tensor(0.0596, grad_fn=<MseLossBackward>)
                policy_losses_loc.append(F.mse_loss(action, probs_loc)*advantage)
                policy_losses_scale.append(F.softplus(probs_scale)*advantage)

                print("policy losses LOC")
                print(policy_losses_loc[-1])
                print("policy losses SCALE")
                print(policy_losses_scale[-1])
                
                # in der Zeile drunter steht der 'alte' value_loss
                # value_losses.append(F.mse_loss(value, torch.tensor([R]))) # tensor([ 0.4873,  0.0398, -0.3586,  0.6294,  0.3986, -0.9495, -0.9261, -0.6679, 0.5093], grad_fn=<UnbindBackward>)
                
                value_losses.append(F.mse_loss(value, torch.tensor([R])))
                print("value Loss")
                print(value_losses[-1])

                # mü = probs_loc und sigma = probs_scale

                # -m.log_prob(action) = tensor([[ nan, -0.5623, -0.4665, 23.3323, -0.0761, nan, nan, nan, nan]], grad_fn=<NegBackward>)
                # m = Normal(loc: torch.Size([9]), scale: torch.Size([9]))

                # IM DISKRETEN FALL: -m.log_prob(action) = tensor(1.4363, grad_fn=<NegBackward>)
                # IM DISKRETEN FALL: m = Categorical(probs: torch.Size([3]), logits: torch.Size([3]))

                '''
                print("M.log_prob")
                print(-m.log_prob(action))
                print("M ___________")
                print(m)
                '''
            # Zeile drunter is die alte loss Summe
            # loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() # mean squared loss
            loss = torch.stack(policy_losses_loc).sum() + torch.stack(policy_losses_scale).sum() + torch.stack(value_losses).sum()
            
            
            print("Loss _________________________")
            print(loss)
            print(loss.item())

            # loss --> tensor(nan, grad_fn=<AddBackward0>)
            # loss.item() --> nan
            

            # Optimize joint loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Don't forget to delete all experiences afterwards! This is an on-policy algorithm.
            self.transitions.clear()

        return loss
