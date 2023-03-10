import random
import math
import mlflow
import numpy
from numpy.core.fromnumeric import shape, std
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import Agent
from .advantages import temporal_difference, advantage_actor_critic, reinforce, nstep

class A2CNet(nn.Module):
    def __init__(self, nr_input_features, nr_actions, activation, nr_hidden_units = 64):
        super(A2CNet, self).__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            A2CLearner.create_activation(activation),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            A2CLearner.create_activation(activation),
        )
        self.action_head_loc = nn.Sequential( # Actor LOC-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Tanh(),
        )
        self.action_head_scale = nn.Sequential( # Actor SCALE-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            #nn.Softplus(),
            nn.Sigmoid(),
        ) # Actor = Policy-Function NN
        self.value_head = nn.Linear(nr_hidden_units, 1) # Critic = Value-Function NN

    def forward(self, x):
        x = self.fc_net(x)
        # x = x.view(x.size(0), -1) # reshapes the tensor
        return (self.action_head_loc(x), self.action_head_scale(x)) ,  self.value_head(x)


class A2CNetSplit(nn.Module):
    def __init__(self, nr_input_features, nr_actions, activation, nr_hidden_units = 64):
        super(A2CNetSplit, self).__init__()
        self.policy_base_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            A2CLearner.create_activation(activation),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            A2CLearner.create_activation(activation),
        )
        self.action_head_loc = nn.Sequential( # Actor LOC-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Tanh(),
        )
        self.action_head_scale = nn.Sequential( # Actor SCALE-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            #nn.Softplus(),
            nn.Sigmoid(),
        ) # Actor = Policy-Function NN

        self.value_head = nn.Sequential( #Critic = Value-Function
            nn.Linear(nr_input_features, nr_hidden_units),
            A2CLearner.create_activation(activation),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            A2CLearner.create_activation(activation),
            nn.Linear(nr_hidden_units,1),
        )

    def forward(self, states):
        x = self.policy_base_net(states)
        # x = x.view(x.size(0), -1) # reshapes the tensor
        return (self.action_head_loc(x), self.action_head_scale(x)) ,  self.value_head(states)


DEFAULT_ALPHA = 0.001
DEFAULT_GAMMA = 0.999
DEFAULT_ENTROPY = 1e-4
DEFAULT_ENTROPY_FALL = 1 
DEFAULT_BATCH_SIZE = 1
DEFAULT_SCALE_CLAMP_MIN = 0.
DEFAULT_SCALE_CLAMP_MAX = 1. 

DEFAULT_NET = "multihead"
DEFAULT_ADVANTAGE = "a2c"
DEFAULT_ACTIVATION_FKT = "ReLu"
DEFAULT_HIDDEN_NEURONS = 64

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
        parser.add_argument('-a','--alpha', type=float, default=DEFAULT_ALPHA, help='Initial learning rate for AdamOptimizer.')
        parser.add_argument('-g','--gamma', type=float, default=DEFAULT_GAMMA , help='Discount factor for rewards, future rewards should be weight less.')
        parser.add_argument('-e', '--entropy_beta', type=float, default=DEFAULT_ENTROPY, help='Factor for entropy term, how strong the policy space is smoothed.')
        parser.add_argument('-ef', '--entropy_fall', type=float, default=DEFAULT_ENTROPY_FALL, help='Entropy decay over time.')
        parser.add_argument('-b', '--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Frequency of gradient application, every n episodes.')
        parser.add_argument('-cmin', '--scale_clamp_min', type=float, default=DEFAULT_SCALE_CLAMP_MIN, help='Minimum of the action scale clamp.')
        parser.add_argument('-cmax', '--scale_clamp_max', type=float, default=DEFAULT_SCALE_CLAMP_MAX, help='Maximum of the action scale clamp.')
        return parser

    @staticmethod
    def add_config_args(parser):
        parser.add_argument('-adv', '--advantage', type=str, default=DEFAULT_ADVANTAGE, choices=["a2c", "td", "3step", "reinforce"], help='The chosen advantage, default a2c.')
        parser.add_argument('-net', '--network', type=str, default=DEFAULT_NET, choices=["split", "multihead"], help='Splithead uses two different nets for actor and critic, Multihead on one.')
        parser.add_argument('-act', '--activation', type=str, default=DEFAULT_ACTIVATION_FKT, choices=["ReLu", "sigmoid", "tanh"], help='Activation function for hidden layers.')
        parser.add_argument('-hn', '--hidden_neurons', type=int, default=DEFAULT_HIDDEN_NEURONS, help="Number neurons in hidden layers.")
        return parser

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
        advantage=DEFAULT_ADVANTAGE, network=DEFAULT_NET, activation=DEFAULT_ACTIVATION_FKT, hidden_neurons=DEFAULT_HIDDEN_NEURONS,
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
            "activation": activation,
            "hidden_neurons": hidden_neurons,
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
            self.a2c_net = A2CNet(self.nr_input_features, self.nr_actions, self.activation, nr_hidden_units=hidden_neurons).to(self.device)
        else:
            self.a2c_net = A2CNetSplit(self.nr_input_features, self.nr_actions, self.activation, nr_hidden_units=hidden_neurons).to(self.device)

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
            torch.tensor([state], device=self.device, dtype=torch.float32)
        )
        m = torch.distributions.normal.Normal(action_locs, action_scales)
        action = m.sample().detach() 
        return action.cpu().numpy() # Shape [1,action_space]


    """
     Predicts the action probabilities.
    """       
    def predict_policy(self, states):
        (action_locs, action_scales), values = self.a2c_net(states)
        # we can clamp the action scales, but do not have to, by setting them to 0 and int max
        action_scales = action_scales.clamp(min=self.scale_clamp_min, max=self.scale_clamp_max) 
        return (action_locs, action_scales), values

    def _discounted_returns(self, rewards):
        # Calculate discounted returns
        discounted_returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma*R
            discounted_returns.append(R)
        discounted_returns.reverse()
        return torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)

    """
     Performs a learning update of the currently learned policy and value function.
    """
    def update(self, nr_episode, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        # 64 Werte fuer state, 9 Werte fuer action, 1 Wert fuer reward, 64 Werte fuer next_state, 1 boolean fuer done

        if done:
            states, actions, rewards, next_states, dones = tuple(zip(*self.transitions))

            returns = self._discounted_returns(rewards)                          # Shape([1000])
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)          # Shape([1000])
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)          # Shape([1000,1,9])
            actions = actions.squeeze(1) # remove n worms dim                               # Shape([1000,9])
            states = torch.tensor(states, device=self.device, dtype=torch.float32)            # Shape[1000,64]
            next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)  # Shape[1000,64]

            (action_locs, action_scales), state_values = self.predict_policy(states)        # Shape[1000,9], Shape[1000,9], Shape[1000,1]
            state_values = state_values.squeeze(1)                                          # Shape[1000]
            _, next_state_values = self.predict_policy(next_states)                         # Shape[1000,1]
            next_state_values = next_state_values.squeeze(1)                                # Shape[1000]

            if self.advantage == "a2c":
                advantages = advantage_actor_critic(Rs=returns,values=state_values,)
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
                advantages = reinforce(Rs=returns,)
            else:
                raise RuntimeError()

            # detaching the advantage is important to avoid any gradient from the policy net to leak over to the value net
            advantages = advantages.detach() #Shape[1000] 

            cur_entropy_beta = self.entropy_beta * (self.entropy_fall ** nr_episode)

            # this is the log prob of the normal distribution
            # (loc - self.loc )*2 / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
            # when confidence in an action rises and scale falls, 
            #   the gradient of var becomes exponentially smaller,
            #   the gradient of loc  becomes linearilly smaller
            # => this may result in unstable behaviour, we could modify the log prob to divide by 2 scale, to get stable gradients
            # => or introduce a lower bound for the scale

            normal_distr = torch.distributions.normal.Normal(action_locs, action_scales, )
            entropy_losses = - cur_entropy_beta * normal_distr.entropy().mean(1) * advantages  # Shape [1000,9] -> Shape [1000] 
            policy_losses = - normal_distr.log_prob(actions).mean(1) * advantages # Shape [1000]

            #value_loss = F.smooth_l1_loss(state_values, returns, reduction='sum')
            value_loss = F.mse_loss(input=state_values, target=returns) #potentially don't do reduction
            #print("value_loss", value_loss.item())

            entropy_loss,  policy_loss, = entropy_losses.sum(), policy_losses.sum(),
            loss = entropy_loss + value_loss + policy_loss
            measures = {
                "loss": loss.item(),
                "loss_policy": policy_loss.item() ,
                "loss_entropy": entropy_loss.item(),
                "loss_value" : value_loss.item(),
                "advantages_std" : advantages.std().item(),
                "advantages_avg" : advantages.mean().item(),
                "action_loc_std": action_locs.std().item(),
                "action_loc_avg": action_locs.mean().item(),
                "action_scale_avg": action_scales.mean().item(),
                "action_scale_std": action_scales.std().item(),
                "state_value_avg": state_values.mean().item(),
                "state_value_std": state_values.std().item(),
                "returns_avg" : returns.mean().item(),
            }

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
