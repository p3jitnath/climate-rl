import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, envs, layer_size):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(envs.single_observation_space.shape).prod()
            + np.prod(envs.single_action_space.shape),
            layer_size,
        )
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
