import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, envs, layer_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(
            envs.single_observation_space.shape[0], layer_size
        )
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc_mean = nn.Linear(
            layer_size, np.prod(envs.single_action_space.shape)
        )
        self.fc_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = self.fc_logstd.exp()
        return mean, std

    def get_action(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, mean
