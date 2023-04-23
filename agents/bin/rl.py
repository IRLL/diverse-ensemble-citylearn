# -*- coding: utf-8 -*-
# @Time    : 2022/7/22 3:07 PM
# @Author  : Zhihu Yang

# from agents.bin.encoder import Encoder

# conditional imports
try:
    import torch
    from torch.distributions import Normal
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise Exception(
        "This functionality requires you to install torch. You can install torch by : pip install torch torchvision, or for more detailed instructions please visit https://pytorch.org.")


class PolicyNetwork(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 action_space,
                 action_scaling_coef,
                 hidden_dim=[400, 300],
                 init_w=3e-3,
                 log_std_min=-20,
                 log_std_max=2,
                 epsilon=1e-6,
                 ):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        self.device = None
        # self.encoder = Encoder(num_inputs, num_inputs, hidden_dim)

        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean_linear = nn.Linear(hidden_dim[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[1], num_actions)

        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = torch.FloatTensor(
            action_scaling_coef * (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            action_scaling_coef * (action_space.high + action_space.low) / 2.)

    def forward(self, state, detach):
        # state = self.encoder(state, detach)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state, detach):
        mean, log_std = self.forward(state, detach)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.device = device
        return super(PolicyNetwork, self).to(device)


class Actor(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_actions,
                 action_space,
                 action_scaling_coef,
                 hidden_dim=[400, 300],
                 init_w=3e-3,
                 log_std_min=-20,
                 log_std_max=2,
                 epsilon=1e-6,
                 encode_dim=None
                 ):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon
        self.device = None
        # self.encode_dim = encode_dim if encode_dim is not None else num_inputs
        # self.encoder = Encoder(num_inputs, self.encode_dim, hidden_dim)

        # self.linear1 = nn.Linear(self.encode_dim, hidden_dim[0])
        self.linear1 = nn.Linear(num_inputs, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[0])
        self.linear3 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean_linear = nn.Linear(hidden_dim[1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[1], num_actions)

        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = torch.FloatTensor(
            action_scaling_coef * (action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor(
            action_scaling_coef * (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, std

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.device = device
        return super(Actor, self).to(device)


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[400, 300], init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[0])
        self.linear3 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear4 = nn.Linear(hidden_size[1], 1)
        self.ln1 = nn.LayerNorm(hidden_size[0])
        self.ln2 = nn.LayerNorm(hidden_size[1])
        self.ln3 = nn.LayerNorm(hidden_size[1])

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        # x = torch.cat([state, action], 1)
        x = self.ln1(F.relu(self.linear1(x)))
        x = self.ln2(F.relu(self.linear2(x)))
        x = self.ln3(F.relu(self.linear3(x)))
        x = self.linear4(x)
        return x


class Critic(nn.Module):
    def __init__(
            self,
            num_inputs,
            num_actions,
            hidden_size,
            encode_dim=None
    ):
        super(Critic, self).__init__()
        # self.encode_dim = encode_dim if encode_dim is not None else num_inputs
        # self.encoder = Encoder(num_inputs, self.encode_dim, hidden_size)
        # self.Q1 = SoftQNetwork(self.encode_dim, num_actions, hidden_size)
        # self.Q2 = SoftQNetwork(self.encode_dim, num_actions, hidden_size)
        self.Q1 = SoftQNetwork(num_inputs, num_actions, hidden_size)
        self.Q2 = SoftQNetwork(num_inputs, num_actions, hidden_size)

    def forward(self, state, action, detach=False):
        # state = self.encoder(state, detach)
        x = torch.cat([state, action], dim=-1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)

        return q1, q2


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
