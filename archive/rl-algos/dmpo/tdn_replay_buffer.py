import warnings
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import psutil
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class TDNReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    bootstrapped_discounts: torch.Tensor


class TDNReplayBuffer(ReplayBuffer):
    """
    We have enhanced Stable Baseline3 to support TD(n) with a new buffer that stores bootstrapped discounts:
    [1] When adding the remaining elements of the n-step rolling buffer after the environment's completion, it's essential to define the integer used as an exponent to lambda for calculating the discount factor. This factor is multiplied with the bootstrapped predicted value from our Q-function. Ordinarily, it equals n-step, but for the leftover elements in the n-step rolling buffer post-environment completion, the horizon is less than n-step, making the exponent strictly less than 0.
    [2] This feature is also advantageous when an episode finishes in less than n-step.
    [3] Upon episode completion, no bootstrapping occurs, which might seem redundant. However, the Acme introduction paper suggests that bootstrapping should happen in cases of episode truncation. While the complexity of the library's modularity made it challenging to confirm this implementation in Acme, we adhere to the paper's guidelines.
    [4] NOTE: The Stable Baseline3 ReplayBuffer necessitates setting the ``TimeLimit.truncated``` field in the info to True for accurate truncation handling, a detail NOT automatically set by Gym, thus might require our intervention.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.bootstrapped_discounts = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

        if psutil is not None:
            mem_available = psutil.virtual_memory().available
            total_memory_usage = self.bootstrapped_discounts.nbytes

            if total_memory_usage > mem_available:
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete (augmented with bootstrapped_discounts)"
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        bootstrapped_discount: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if isinstance(infos, list):
            infos = [infos]
        self.bootstrapped_discounts[self.pos] = np.array(
            bootstrapped_discount
        ).copy()  # Important to set before super, because super increases self.pos
        super().add(obs, next_obs, actions, rewards, dones, infos)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> TDNReplayBufferSamples:
        env_indices = np.random.randint(
            0, high=self.n_envs, size=(len(batch_inds),)
        )  # Sample randomly the env idx

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[
                    (batch_inds + 1) % self.buffer_size, env_indices, :
                ],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(
                self.observations[batch_inds, env_indices, :], env
            ),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
            self.bootstrapped_discounts[batch_inds, env_indices].reshape(
                -1, 1
            ),
        )
        return TDNReplayBufferSamples(*tuple(map(self.to_torch, data)))
