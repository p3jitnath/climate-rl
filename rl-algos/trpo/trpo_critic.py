import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# from trpo_utils import layer_init


class Critic(nn.Module):
    def __init__(self, envs, layer_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(
                np.array(envs.single_observation_space.shape).prod(),
                layer_size,
            ),
            nn.Tanh(),
            nn.Linear(layer_size, layer_size),
            nn.Tanh(),
            nn.Linear(layer_size, 1),
        )

    def get_value(self, x):
        return self.critic(x)
