import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import layer_init, uniform_scaling_layer_init


class Critic(nn.Module):
    def __init__(self, env, categorical_num_bins):
        super().__init__()
        self.value_linear_1 = uniform_scaling_layer_init(
            nn.Linear(
                env.single_observation_space.shape[0]
                + env.single_action_space.shape[0],
                256,
            )
        )
        self.layer_norm = nn.LayerNorm(256)

        self.value_linear_2 = uniform_scaling_layer_init(nn.Linear(256, 256))
        self.value_linear_3 = uniform_scaling_layer_init(nn.Linear(256, 256))
        self.value_linear_4 = layer_init(
            nn.Linear(256, categorical_num_bins),
            std=1e-5,
            variance_scaling=True,
        )

    def forward(self, x, a):
        a = torch.clip(a, -1, 1)
        x = torch.cat([x, a], -1)

        h = self.value_linear_1(x)
        h = torch.tanh(self.layer_norm(h))

        h = F.elu(self.value_linear_2(h))
        h = F.elu(self.value_linear_3(h))
        v = self.value_linear_4(h)

        return v
