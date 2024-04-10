import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, envs):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(envs.single_observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3_mean = nn.Linear(128, np.prod(envs.single_action_space.shape))
        self.fc3_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        std = self.fc3_logstd.exp()
        return mean, std

    def get_action(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, mean
