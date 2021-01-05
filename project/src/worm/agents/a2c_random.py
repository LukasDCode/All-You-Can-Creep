import random
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .a2c import A2CLearner


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
        self.action_head_loc = nn.Sequential(  # Actor LOC-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Tanh(),
        )
        self.action_head_scale = nn.Sequential(  # Actor SCALE-Ausgabe von Policy
            nn.Linear(nr_hidden_units, nr_actions),
            nn.Softplus(),
        )  # Actor = Policy-Function NN
        self.value_head = nn.Linear(nr_hidden_units, 1)  # Critic = Value-Function NN

    def forward(self, x):
        x = self.fc_net(x)
        # x = x.view(x.size(0), -1) # reshapes the tensor
        return (self.action_head_loc(x), self.action_head_scale(x)), self.value_head(x)

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
        self.fc_net.load_state_dict(state_dict["fc_net"], strict=strict, )
        self.action_head_loc.load_state_dict(state_dict["action_head_loc"], )
        self.action_head_scale.load_state_dict(state_dict["action_head_scale"], )
        self.value_head.load_state_dict(state_dict["value_head"], strict=strict, )
        return self


"""
 Autonomous agent using Synchronous Actor-Critic.
"""


class A2CLearner_random(A2CLearner):

    def state_dict(self):
        return self.a2c_net.state_dict()

    def load_state_dict(self, state_dict, strict=False):
        self.a2c_net.load_state_dict(state_dict, strict=strict, )

    def __init__(self, params):
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.alpha = params["alpha"]
        self.entropy = params["entropy"]
        self.nr_input_features = params["nr_input_features"]
        self.transitions = []
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        self.a2c_net = A2CNet(self.nr_input_features, self.nr_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.a2c_net.parameters(), lr=params["alpha"])
        self.distance_index_of_observation = 4


    def policy(self, state):
        (action_locs, action_scales), _ = self.predict_policy([state])
        # print("Actions {} {}".format( action_locs, action_scales))
        random_m = torch.rand(9,dtype= float)
        random_m = [random.choice(-i, i) for i in random_m]
        return random_m.sample().detach().cpu().numpy()

