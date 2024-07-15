import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, envs, layer_size):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(envs.single_observation_space.shape).prod(), layer_size
        )
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc_mean = nn.Linear(
            layer_size, np.prod(envs.single_action_space.shape)
        )
        self.fc_logstd = nn.Linear(
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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # from SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x_t = (
            normal.rsample()
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(
            self.action_scale * (1 - y_t.pow(2)) + 1e-6
        )  # enforcing action bound
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
