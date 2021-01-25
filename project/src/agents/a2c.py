import random
import math
import mlflow
import numpy
from numpy.core.fromnumeric import shape, std
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from .agent import Agent
from .advantages import temporal_difference, advantage_actor_critic, reinforce, nstep

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
            nn.Tanh(),
        )
        self.action_head_scale = nn.Sequential( # Actor SCALE-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Softplus(),
        ) # Actor = Policy-Function NN
        self.value_head = nn.Linear(nr_hidden_units, 1) # Critic = Value-Function NN

    def forward(self, x):
        x = self.fc_net(x)
        # x = x.view(x.size(0), -1) # reshapes the tensor
        return (self.action_head_loc(x), self.action_head_scale(x)) ,  self.value_head(x)

    # save with  torch.save(model.state_dict(), path)
    def state_dict(self):
        state_dict = {
            "fc_net": self.fc_net.state_dict(),
            "action_head_loc": self.action_head_loc.state_dict(),
            "action_head_scale": self.action_head_scale.state_dict(),
            "value_head": self.value_head.state_dict(),
        }
        return state_dict

    # load with model.load_state_dict(torch.load(path))
    def load_state_dict(self, state_dict, strict=False):
        self.fc_net.load_state_dict(state_dict["fc_net"], strict=strict,)
        self.action_head_loc.load_state_dict(state_dict["action_head_loc"],)
        self.action_head_scale.load_state_dict(state_dict["action_head_scale"],)
        self.value_head.load_state_dict(state_dict["value_head"], strict=strict, )
        return self


class A2CNetSplit(nn.Module):
    def __init__(self, nr_input_features, nr_actions):
        super(A2CNetSplit, self).__init__()
        nr_hidden_units = 64
        self.policy_base_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU()
        )
        self.action_head_loc = nn.Sequential( # Actor LOC-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Tanh(),
        )
        self.action_head_scale = nn.Sequential( # Actor SCALE-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Softplus(),
        ) # Actor = Policy-Function NN

        self.value_head = nn.Sequential( #Critic = Value-Function
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units,1),
        )

    def forward(self, states):
        x = self.policy_base_net(states)
        # x = x.view(x.size(0), -1) # reshapes the tensor
        return (self.action_head_loc(x), self.action_head_scale(x)) ,  self.value_head(states)

    # save with  torch.save(model.state_dict(), path)
    def state_dict(self):
        state_dict = {
            "policy_base_net": self.policy_base_net.state_dict(),
            "action_head_loc": self.action_head_loc.state_dict(),
            "action_head_scale": self.action_head_scale.state_dict(),
            "value_head": self.value_head.state_dict(),
        }
        return state_dict

    # load with model.load_state_dict(torch.load(path))
    def load_state_dict(self, state_dict, strict=False):
        self.policy_base_net.load_state_dict(state_dict["policy_base_net"], strict=strict,)
        self.action_head_loc.load_state_dict(state_dict["action_head_loc"], strict=strict,)
        self.action_head_scale.load_state_dict(state_dict["action_head_scale"], strict=strict)
        self.value_head.load_state_dict(state_dict["value_head"], strict=strict, )
        return self


DEFAULT_ALPHA = 0.001
DEFAULT_GAMMA = 0.999
DEFAULT_ENTROPY = 1e-4
DEFAULT_ENTROPY_FALL = 0.999
DEFAULT_BATCH_SIZE = 10
DEFAULT_SCALE_CLAMP_MIN = 0.01
DEFAULT_SCALE_CLAMP_MAX = 1.

DEFAULT_NET = "multihead"
DEFAULT_ADVANTAGE = "a2c"

"""
 Autonomous agent using Synchronous Actor-Critic.
"""
class A2CLearner(Agent):

    @staticmethod
    def agent_name():
        return "a2c"

    @staticmethod
    def hyper_params():
        return {
            "alpha": {"min": 0.001, "max": 0.001},
            "gamma": {"min": 0.99, "max": 1.},
            "entropy_beta": {"min": 1e-6, "max": 1},
            "entropy_fall": {"min": 0.99, "max": 1},
            "batch_size": {"min": 10, "max": 10},
            "scale_clamp_min": {"min": 0.01, "max": 0.01},
            "scale_clamp_max": {"min": 1, "max": 1},
        }

    @staticmethod
    def add_hyper_param_args(parser):
        parser.add_argument('-a','--alpha', type=float, default=DEFAULT_ALPHA, help='the learning rate')
        parser.add_argument('-g','--gamma', type=float, default=DEFAULT_GAMMA , help='the discount factor for rewards')
        parser.add_argument('-e', '--entropy_beta', type=float, default=DEFAULT_ENTROPY, help='the exploitation rate')
        parser.add_argument('-ef', '--entropy_fall', type=float, default=DEFAULT_ENTROPY_FALL, help='the entropy decay')
        parser.add_argument('-b', '--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='the batch size')
        parser.add_argument('-cmin', '--scale_clamp_min', type=int, default=DEFAULT_SCALE_CLAMP_MIN, help='the minimum of the action scale clamp')
        parser.add_argument('-cmax', '--scale_clamp_max', type=int, default=DEFAULT_SCALE_CLAMP_MAX, help='the maximum of the action scale clamp')
        return parser

    @staticmethod
    def add_config_args(parser):
        parser.add_argument('-adv', '--advantage', type=str, default=DEFAULT_ADVANTAGE, choices=["a2c", "td", "3step", "reinforce"])
        parser.add_argument('-net', '--network', type=str, default=DEFAULT_NET, choices=["split", "multihead"])
        return parser

    def state_dict(self):
        return {
            "model": self.a2c_net.state_dict(),
            "config":  self.config,
            "state": {k: self.__dict__.get(k) for k in self.config}
        }

    def __init__(
        self,
        env,
        # hyper params
        gamma=DEFAULT_GAMMA, alpha=DEFAULT_ALPHA, entropy_beta=DEFAULT_ENTROPY,
        entropy_fall=DEFAULT_ENTROPY_FALL, batch_size=DEFAULT_BATCH_SIZE,
        scale_clamp_min=DEFAULT_SCALE_CLAMP_MIN, scale_clamp_max=DEFAULT_SCALE_CLAMP_MAX,
        # agent config
        advantage=DEFAULT_ADVANTAGE, network=DEFAULT_NET,
        # Loading model
        only_model=False, state_dict = None,
        **kwargs,
        ):
        super().__init__(env)
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        if state_dict:
            state_dict = state_dict["agent"]

        """Params from constructor"""
        self.config = {
            "gamma":gamma,
            "alpha":alpha,
            "entropy_beta":entropy_beta,
            "entropy_fall": entropy_fall,
            "batch_size": batch_size,
            "scale_clamp_min": scale_clamp_min,
            "scale_clamp_max": scale_clamp_max,
            "network": network,
            "advantage": advantage,
        }
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

        """Create network"""
        if(network == "multihead"):
            self.a2c_net = A2CNet(self.nr_input_features, self.nr_actions).to(self.device)
        else:
            self.a2c_net = A2CNetSplit(self.nr_input_features, self.nr_actions).to(self.device)

        """Load state dict into model"""
        if state_dict:
            print("Loading model...")
            self.a2c_net.load_state_dict(state_dict["model"])
            print("Loaded model.")

        self.optimizer = torch.optim.Adam(self.a2c_net.parameters(), lr=self.alpha)
        self.transitions = []

        """Log Params"""
        print(f"Loaded A2CLearner {str(self.config)}")
        mlflow.log_params({
            "agent": "a2c",
            **self.config,
        })

    """
     Samples a new action using the policy network.
    """
    def policy(self, state):
        (action_locs, action_scales), _ = self.predict_policy(
            torch.tensor([state], device=self.device, dtype=torch.float)
        )
        # print("Actions {} {}".format( action_locs, action_scales))
        m = torch.distributions.normal.Normal(action_locs, action_scales)
        action = m.sample().detach() # Size([1,9])
        return action.cpu().numpy()


    """
     Predicts the action probabilities.
    """       
    def predict_policy(self, states):
        (action_locs, action_scales), values = self.a2c_net(states)
        action_scales = action_scales.clamp(min=self.scale_clamp_min, max=self.scale_clamp_max)
        return (action_locs, action_scales), values
        
    """
     Predicts the state values.
    """       
    def predict_value(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        return self.value_net(states)

    def _normalized_returns(self, rewards):
        # Calculate and normalize discounted returns

        discounted_returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma*R
            discounted_returns.append(R)
        discounted_returns.reverse()

        discounted_returns = torch.tensor(discounted_returns, device=self.device, dtype=torch.float).detach()
        return F.normalize(discounted_returns, dim=0)

    """
     Performs a learning update of the currently learned policy and value function.
    """
    def update(self, nr_episode, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        # 64 Werte fuer state, 9 Werte fuer action, 1 Wert fuer reward, 64 Werte fuer next_state, 1 boolean fuer done

        if done:
            states, actions, rewards, next_states, dones = tuple(zip(*self.transitions))

            normalized_returns = self._normalized_returns(rewards)                          # Shape([1000])
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)          # Shape([1000])
            actions = torch.tensor(actions, device=self.device, dtype=torch.float)          # Shape([1000,1,9])
            actions = actions.squeeze(1) # remove n worms dim                               # Shape([1000,9])
            states = torch.tensor(states, device=self.device, dtype=torch.float)            # Shape[1000,64]
            next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)  # Shape[1000,64]

            (action_locs, action_scales), state_values = self.predict_policy(states)        # Shape[1000,9], Shape[1000,9], Shape[1000,1]
            state_values = state_values.squeeze(1)                                          # Shape[1000]
            state_values = F.normalize(state_values, dim=0, eps=self.eps)
            _, next_state_values = self.predict_policy(next_states)                         # Shape[1000,1]
            next_state_values = next_state_values.squeeze(1)                                # Shape[1000]
            next_state_values = F.normalize(state_values, dim=0, eps=self.eps)

            if self.advantage == "a2c":
                advantages = advantage_actor_critic(Rs=normalized_returns,values=state_values,)
            elif self.advantage == "3step":
                advantages = nstep(
                    n=3,
                    gamma=self.gamma,
                    rewards=rewards,
                    values=state_values,
                    next_values=next_state_values,
                )
            elif self.advantage == "td":
                advantages = temporal_difference(
                    gamma=self.gamma,
                    rewards=rewards,
                    values=state_values,
                    next_values=next_state_values,
                )
            elif self.advantage == "reinforce":
                advantages = reinforce(Rs=normalized_returns,)
            else:
                raise RuntimeError()

            advantages = advantages # Shape[1000] 
            cur_entropy_beta = self.entropy_beta * (self.entropy_fall ** nr_episode)

            normal_distr = torch.distributions.normal.Normal(action_locs, action_scales, )
            entropy_losses = - cur_entropy_beta * normal_distr.entropy().mean(1) * advantages  # Shape [1000,9] -> Shape [1000] 
            policy_losses = - normal_distr.log_prob(actions).mean(1) * advantages # Shape [1000]
            value_loss = F.mse_loss(state_values, normalized_returns)


            entropy_loss,  policy_loss, = entropy_losses.sum(), policy_losses.sum(),
            loss = entropy_loss + value_loss + policy_loss
            measures = {
                "loss": loss.item(),
                "loss_policy": policy_loss.item() ,
                "loss_entropy": entropy_loss.item(),
                "loss_value" : value_loss.item(),
                "advantages_std" : advantages.std().item(),
                "action_loc_std": action_locs.std().item(),
                "action_scale_avg": action_scales.mean().item(),
                "action_scale_std": action_scales.std().item(),
                "state_value_avg": state_values.mean().item(),
                "state_value_std": state_values.std().item(),
            }
            #print(rewards.sum().cpu().item())
            #print(measures["loss_policy"], measures["loss_entropy"],)
            #print(measures["action_scale_avg"],measures["action_scale_avg"])

            # Optimize joint batch loss
            o_step = self.batch_size 
            if nr_episode % o_step == 0: # reset grad at 0, 10, 20...
                self.optimizer.zero_grad()
            loss.backward()
            if (nr_episode + 1) % o_step == 0: # step grad at 9,19,29... 
                self.optimizer.step()
            
            # Don't forget to delete all experiences afterwards! This is an on-policy algorithm.
            self.transitions.clear()

            return measures

        return {}
