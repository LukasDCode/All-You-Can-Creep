import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import mlflow

from .agent import Agent

class PPOMemory:
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

class ActorNetwork(nn.Module):
    """
    Actor Network has 1 base NN with 2 different heads.
    One head returns the location of the Normal distribution
    the other head returns the scale of the Normal distribution.
    """
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=512, fc2_dims=512, actor_scale_head='softplus'):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
        )
        self.action_head_loc = nn.Sequential(
            nn.Linear(fc2_dims, n_actions),
            nn.Tanh(),
        )

        self.action_head_scale = None
        if actor_scale_head == 'softmax':
            self.action_head_scale = nn.Sequential(
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1),
            )   
        elif actor_scale_head == 'softplus':
            self.action_head_scale = nn.Sequential(
                nn.Linear(fc2_dims, n_actions),
                nn.Softplus(),
            )
        elif actor_scale_head == 'sigmoid':
            self.action_head_scale = nn.Sequential(
                nn.Linear(fc2_dims, n_actions),
                nn.Sigmoid(),
            )
        else:
            raise Exception("Invalid scale activation function given.")

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        intermediate_step = self.actor(state)
        locs = self.action_head_loc(intermediate_step)
        scales = self.action_head_scale(intermediate_step)
        return locs, scales

    def state_dict(self):
        state_dict = {
            "actor": self.actor.state_dict(),
            "action_head_loc": self.action_head_loc.state_dict(),
            "action_head_scale": self.action_head_scale.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        self.actor.load_state_dict(state_dict["actor"], strict=strict,)
        self.action_head_loc.load_state_dict(state_dict["action_head_loc"], strict=strict,)
        self.action_head_scale.load_state_dict(state_dict["action_head_scale"], strict=strict)
        return self


class CriticNetwork(nn.Module):
    """
    Critic Network returns 1 number evaluating the passed on state.
    """
    def __init__(self, input_dims, alpha, fc1_dims=512, fc2_dims=512):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def state_dict(self):
        state_dict = {
            "critic": self.critic.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        self.critic.load_state_dict(state_dict["critic"], strict=strict,)
        return self

DEFAULT_ALPHA_ACTOR = 0.0003
DEFAULT_ALPHA_CRITIC = 0.0003
DEFAULT_BATCH_SIZE = 2024
DEFAULT_BUFFER_SIZE = 20240
DEFAULT_EPOCHS = 3
DEFAULT_GAMMA = 0.995
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_POLICY_CLIP = 0.2

DEFAULT_HIDDEN_ACTOR = 512
DEFAULT_HIDDEN_CRITIC = 512
DEFAULT_ACTOR_SCALE_HEAD = 'softplus'

class PPOv2Learner(Agent):

    @staticmethod
    def agent_name():
        return "PPOv2"
    
    @staticmethod
    def add_hyper_param_args(parser):
        parser.add_argument('-aa', '--alpha_actor', type=float, default=DEFAULT_ALPHA_ACTOR, help='Learning rate actor NN')
        parser.add_argument('-ac', '--alpha_critic', type=float, default=DEFAULT_ALPHA_CRITIC, help='Learning rate critic NN')
        parser.add_argument('-bas', '--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
        parser.add_argument('-bus', '--buffer_size', type=int, default=DEFAULT_BUFFER_SIZE, help='Buffer size, should be multiple batch size')
        parser.add_argument('-epochs','--n_epochs', type=int, default=DEFAULT_EPOCHS, help='Epochs of updates')
        parser.add_argument('-g','--gamma', type=float, default=DEFAULT_GAMMA, help='Discount factor for rewards')
        parser.add_argument('-gae','--gae_lambda', type=float, default=DEFAULT_GAE_LAMBDA, help='Diminishing Gamma in discounting')
        parser.add_argument('-clip','--policy_clip', type=float, default=DEFAULT_POLICY_CLIP, help='TODO')
        return parser

    @staticmethod
    def add_config_args(parser):
        parser.add_argument('-ha', '--hidden_actor', type=int, default=DEFAULT_HIDDEN_ACTOR, help='hidden layer units actor NN')
        parser.add_argument('-hc', '--hidden_critic', type=int, default=DEFAULT_HIDDEN_CRITIC, help='hidden layer units critic NN')
        parser.add_argument('-ash', '--actor_scale_head', type=str, default=DEFAULT_ACTOR_SCALE_HEAD, help='activation function for scale head from actor NN')
        return parser

    def state_dict(self):
        return {
            "actor_model": self.actor.state_dict(),
            "critic_model": self.critic.state_dict(),
            "config":  self.config,
            "state": {k: self.__dict__.get(k) for k in self.config}
        }

    def __init__(

        self,
        env,

        # measurements dict that is returned every episode
        current_measures = {},

        # hyper params
        alpha_actor = DEFAULT_ALPHA_ACTOR,
        alpha_critic = DEFAULT_ALPHA_CRITIC,
        batch_size = DEFAULT_BATCH_SIZE,
        buffer_size = DEFAULT_BUFFER_SIZE,
        n_epochs = DEFAULT_EPOCHS,
        gamma = DEFAULT_GAMMA,
        gae_lambda = DEFAULT_GAE_LAMBDA,
        policy_clip = DEFAULT_POLICY_CLIP,

        # config params
        hidden_actor=DEFAULT_HIDDEN_ACTOR,
        hidden_critic=DEFAULT_HIDDEN_CRITIC,
        actor_scale_head=DEFAULT_ACTOR_SCALE_HEAD,

        # Loading model
        only_model=False, state_dict = None,
        **kwargs):

        super().__init__(env)

        self.memory = PPOMemory(batch_size)

        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.batch_size = batch_size,
        self.buffer_size = buffer_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip

        self.hidden_actor = hidden_actor
        self.hidden_critic = hidden_critic
        self.actor_scale_head = actor_scale_head
        
        self.current_measures = current_measures

        if state_dict:
            state_dict = state_dict["agent"]

        """Params from constructor, get stored in the saved model"""
        self.config = {
            "alpha_actor" : alpha_actor,
            "alpha_critic" : alpha_critic,
            "batch_size" : batch_size,
            "buffer_size" : buffer_size,
            "n_epochs" : n_epochs,
            "gamma" : gamma,
            "gae_lambda" : gae_lambda,
            "policy_clip" : policy_clip,
            "hidden_actor" : hidden_actor,
            "hidden_critic" : hidden_critic,
            "actor_scale_head" : actor_scale_head,
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

        self.actor = ActorNetwork(self.nr_actions, self.nr_input_features, self.alpha_actor, fc1_dims=self.hidden_actor, fc2_dims=self.hidden_actor, actor_scale_head=self.actor_scale_head)
        self.critic = CriticNetwork(self.nr_input_features, self.alpha_critic, fc1_dims=self.hidden_critic, fc2_dims=self.hidden_critic)

        """Load state dict into model"""
        if state_dict:
            print("Loading model...")
            self.actor.load_state_dict(state_dict["actor_model"])
            self.critic.load_state_dict(state_dict["critic_model"])
            print("Loaded model.")

        """Log Params"""
        print(f"Loaded PPOv2Learner {str(self.config)}")
        mlflow.log_params({
            "agent": "ppo_v2",
            **self.config,
        })

    def get_measures(self):
        return self.current_measures

    def get_buffersize(self):
        return self.buffer_size

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def policy(self, state):

        state = T.tensor([state], dtype=T.float).to(self.actor.device)

        locs, scales = self.actor(state)
        value = self.critic(state)

        dist = T.distributions.normal.Normal(locs, scales)
        action = dist.sample()
        
        probs = T.squeeze(dist.log_prob(action)).tolist()
        action = T.squeeze(action).tolist()
        value = T.squeeze(value).item()

        return action, probs, value

    def update(self):

        actor_loss_list, critic_loss_list, total_loss_list = [], [], []
        loc_list, scale_list = [], []

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            a = np.zeros(len(reward_arr), dtype=np.float32)

            for i in range(len(reward_arr)-1):
                a[-(i+2)] = a[-(i+1)] * self.gamma * self.gae_lambda + reward_arr[-(i+2)] + self.gamma*values[-(i+1)]*(1-int(dones_arr[-(i+2)])) - values[-(i+2)]

            advantage = T.tensor(a).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                locs, scales = self.actor(states)
                dist = T.distributions.normal.Normal(locs, scales)

                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)

                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch].unsqueeze(1) * prob_ratio
                
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch].unsqueeze(1)
                
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + 0.5*critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                loc_list.append(locs.mean().item())
                scale_list.append(scales.mean().item())

        measures = {
            "total_loss": sum(total_loss_list)/len(total_loss_list),
            "actor_loss": sum(actor_loss_list)/len(actor_loss_list),
            "critic_loss": sum(critic_loss_list)/len(critic_loss_list),

            "loc": sum(loc_list)/len(loc_list),
            "scale": sum(scale_list)/len(scale_list),
        }
        self.current_measures = measures

        self.memory.clear_memory()
        return measures
