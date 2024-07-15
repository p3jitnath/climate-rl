import numpy as np
import torch
import torch.nn as nn


class QuantileCritics(nn.Module):
    def __init__(self, envs, n_quantiles, n_critics, layer_size):
        super().__init__()
        state_dim = np.prod(envs.single_observation_space.shape)
        action_dim = np.prod(envs.single_action_space.shape)
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.n_total_quantiles = n_quantiles * n_critics

        def make_critic():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, layer_size),
                nn.ReLU(),
                nn.Linear(layer_size, layer_size),
                nn.ReLU(),
                nn.Linear(layer_size, n_quantiles),
            )

        self.critics = nn.ModuleList([make_critic() for _ in range(n_critics)])

    def forward(self, states, actions):
        state_actions = torch.cat([states, actions], dim=-1)
        return torch.stack(
            tuple(critic(state_actions) for critic in self.critics),
            dim=1,
        )
