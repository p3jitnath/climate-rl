import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import layer_init, uniform_scaling_layer_init


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.linear_1 = uniform_scaling_layer_init(
            nn.Linear(env.single_observation_space.shape[0], 256)
        )
        self.layer_norm = nn.LayerNorm(256)

        self.linear_2 = uniform_scaling_layer_init(nn.Linear(256, 256))
        self.linear_3 = uniform_scaling_layer_init(nn.Linear(256, 256))

        self._mean_layer = layer_init(
            nn.Linear(256, env.single_action_space.shape[0]),
            std=1e-4,
            variance_scaling=True,
        )
        self._stddev_layer = layer_init(
            nn.Linear(256, env.single_action_space.shape[0]),
            std=1e-4,
            variance_scaling=True,
        )

    def forward(self, x, policy_init_scale=0.5, policy_min_scale=1e-6):
        h = self.linear_1(x)
        h = torch.tanh(self.layer_norm(h))

        h = F.elu(self.linear_2(h))
        h = F.elu(self.linear_3(h))

        mean = self._mean_layer(h)
        stddev = F.softplus(self._stddev_layer(h))

        stddev *= policy_init_scale / F.softplus(
            torch.zeros(1, device=x.device)
        )
        stddev += policy_min_scale

        return mean, stddev
