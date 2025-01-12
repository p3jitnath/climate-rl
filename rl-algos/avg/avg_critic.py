import numpy as np
import torch
import torch.nn as nn


def orthogonal_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class Critic(nn.Module):
    def __init__(self, envs, layer_size):
        super(Critic, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(
                np.array(envs.single_observation_space.shape).prod()
                + np.prod(envs.single_action_space.shape),
                layer_size,
            ),
            nn.LeakyReLU(),
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU(),
        )
        self.q = nn.Linear(layer_size, 1)
        self.apply(orthogonal_weight_init)

    def forward(self, x, a):
        x = torch.cat((x, a), -1)
        phi = self.phi(x)
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))
        return self.q(phi).view(-1)
