import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def orthogonal_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, envs, layer_size):
        super(Actor, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(
                np.array(envs.single_observation_space.shape).prod(),
                layer_size,
            ),
            nn.LeakyReLU(),
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(
            layer_size, np.prod(envs.single_action_space.shape)
        )
        self.log_std = nn.Linear(
            layer_size, np.prod(envs.single_action_space.shape)
        )

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.action_space.high - envs.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.action_space.high + envs.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

        self.apply(orthogonal_weight_init)

    def forward(self, x):
        phi = self.phi(x)
        phi = phi / torch.norm(phi, dim=1).view((-1, 1))
        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        dist = torch.distributions.MultivariateNormal(
            mu, torch.diag_embed(log_std.exp())
        )
        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)
        lprob -= (
            2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))
        ).sum(axis=1)
        lprob -= torch.log(self.action_scale).sum()

        # N.B: Tanh must be applied _only_ after lprob estimation of dist sampled action!!
        #      A mistake here can break learning :/
        action = torch.tanh(action_pre) * self.action_scale + self.action_bias
        action_info = {
            "mu": mu,
            "log_std": log_std,
            "dist": dist,
            "lprob": lprob,
            "action_pre": action_pre,
        }

        return action, action_info
