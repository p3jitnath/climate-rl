import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from dmpo_actor import Actor
from dmpo_critic import Critic
from tdn_replay_buffer import TDNReplayBuffer
from torch.utils.tensorboard import SummaryWriter

os.environ["MUJOCO_GL"] = "egl"


@dataclass
class Args:
    exp_name: str = "dmpo_torch"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "personal-p3jitnath"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = int(1e6) + 1
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    epsilon_non_parametric: float = 0.1
    """KL regularization coefficient for the non-parametric distribution"""
    epsilon_parametric_mu: float = 0.01
    """KL regularization coefficient for the mean of the parametric policy distribution"""
    epsilon_parametric_sigma: float = 1e-6
    """KL regularization coefficient for the std of the parametric policy distribution"""
    epsilon_penalty: float = 0.001
    """KL regularization coefficient for the action limit penalty"""
    target_network_update_period: float = 100
    """number of steps before updating the target networks"""
    variable_update_period: int = 1000
    """number of steps before updating the environment interaction actor network"""
    action_sampling_number: float = 20
    """number of actions to sample for each state sampled, to compute an approximated non-parametric better policy distribution"""
    grad_norm_clip: float = 40.0
    """gradients norm clipping coefficient"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_q_lr: float = 3e-4
    """the learning rate of the policy network and critic network optimizer"""
    dual_lr: float = 1e-2
    """the learning rate of the dual parameters"""
    policy_init_scale: float = 0.5
    """scaling coefficient of the policy std"""
    policy_min_scale: float = 1e-6
    """scalar to add to the scaled std of the policy"""
    n_step: float = 4
    """horizon for bootstrapping the target q-value"""
    vmax_values: float = 1600.0
    """max support of the random variable of which the q function will predict the distribution"""
    categorical_num_bins: float = 51
    """number elements in the support of the random variable"""


_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


args = tyro.cli(Args)
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device(
    "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
)

# 0. env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
)
assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"

actor = Actor(envs).to(device)
target_actor = Actor(envs).to(device)
target_actor.load_state_dict(actor.state_dict())
env_actor = Actor(envs).to(device)
env_actor.load_state_dict(target_actor.state_dict())

qf = Critic(envs, args.categorical_num_bins).to(device)
target_qf = Critic(envs, args.categorical_num_bins).to(device)
target_qf.load_state_dict(qf.state_dict())

log_eta = torch.tensor([10.0], requires_grad=True, device=device)
log_alpha_mean = torch.tensor(
    [10.0] * envs.single_action_space.shape[0],
    requires_grad=True,
    device=device,
)
log_alpha_stddev = torch.tensor(
    [1000.0] * envs.single_action_space.shape[0],
    requires_grad=True,
    device=device,
)
log_penalty_temperature = torch.tensor(
    [10.0], requires_grad=True, device=device
)

envs.single_observation_space.dtype = np.float32
rb = TDNReplayBuffer(
    args.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=True,
)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.policy_q_lr)
critic_optimizer = torch.optim.Adam(qf.parameters(), lr=args.policy_q_lr)
dual_optimizer = torch.optim.Adam(
    [log_eta, log_alpha_mean, log_alpha_stddev, log_penalty_temperature],
    lr=args.dual_lr,
)

n_step_obs_rolling_buffer = np.zeros(
    (args.n_step,) + envs.single_observation_space.shape
)
n_step_action_rolling_buffer = np.zeros(
    (args.n_step,) + envs.single_action_space.shape
)
n_step_reward_rolling_buffer = np.zeros((args.n_step,))
n_step_gammas = args.gamma ** np.arange(args.n_step)

v_max = args.vmax_values
v_min = -v_max
rd_var_support = torch.linspace(
    v_min, v_max, steps=args.categorical_num_bins, device=device
)

step_since_last_terminated = 0
sgd_steps = 0

start_time = time.time()

# 1. start the game
obs, _ = envs.reset(seed=args.seed)
for global_step in range(args.total_timesteps):
    with torch.no_grad():
        # 2. retrieve action(s)
        taus_mean, taus_stddev = env_actor(torch.Tensor(obs).to(device))
        distribution = (
            torch.distributions.multivariate_normal.MultivariateNormal(
                loc=taus_mean, scale_tril=torch.diag_embed(taus_stddev)
            )
        )
        taus = distribution.sample().cpu()

    # 3. execute the game and log data
    next_obs, rewards, terminations, truncations, infos = envs.step(
        taus.numpy().clip(-1, 1)
    )

    n_step_obs_rolling_buffer = np.concatenate(
        [n_step_obs_rolling_buffer[1:], obs], 0
    )
    n_step_action_rolling_buffer = np.concatenate(
        [n_step_action_rolling_buffer[1:], taus], 0
    )
    n_step_reward_rolling_buffer = np.concatenate(
        [n_step_reward_rolling_buffer[1:], rewards], 0
    )

    # 4. save data to replay buffer
    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
        if trunc:
            real_next_obs[idx] = infos["final_observation"][idx]

    if step_since_last_terminated > args.n_step:
        n_step_discounted_reward = (
            n_step_reward_rolling_buffer * n_step_gammas
        ).sum()
        rb.add(
            n_step_obs_rolling_buffer[0],
            real_next_obs,
            n_step_action_rolling_buffer[0],
            n_step_discounted_reward,
            terminations,
            np.ones((1,)) * args.n_step,
            infos,
        )
    else:
        n_step_discounted_reward = (
            n_step_reward_rolling_buffer[
                args.n_step - 1 - step_since_last_terminated :
            ]
            * n_step_gammas[: step_since_last_terminated + 1]
        ).sum()
        rb.add(
            n_step_obs_rolling_buffer[
                args.n_step - 1 - step_since_last_terminated
            ],
            real_next_obs,
            n_step_action_rolling_buffer[
                args.n_step - 1 - step_since_last_terminated
            ],
            n_step_discounted_reward,
            terminations,
            np.ones((1,)) * (step_since_last_terminated + 1),
            infos,
        )

    step_since_last_terminated += 1
    obs = next_obs

    if "final_info" in infos:
        for info in infos["final_info"]:
            if step_since_last_terminated > args.n_step:
                # I. Case where rolling_buffer was filled (env ends after n_step)
                # and therefore first entry of the rolling buffer has already been dealt
                for i in range(1, args.n_step):
                    n_step_discounted_reward = (
                        n_step_reward_rolling_buffer[i:] * n_step_gammas[:-i]
                    ).sum()
                    rb.add(
                        n_step_obs_rolling_buffer[i],
                        real_next_obs,
                        n_step_action_rolling_buffer[i],
                        n_step_discounted_reward,
                        terminations,
                        np.ones((1,)) * (args.n_step - i),
                        infos,
                    )
            else:
                # II. Case where env ends before n_step
                # and therefore the first entry(s) wasn't dealt with
                for i in range(0, step_since_last_terminated):
                    n_step_discounted_reward = (
                        n_step_reward_rolling_buffer[i:] * n_step_gammas[:-i]
                    ).sum()
                    rb.add(
                        n_step_obs_rolling_buffer[i],
                        real_next_obs,
                        n_step_action_rolling_buffer[i],
                        n_step_discounted_reward,
                        terminations,
                        np.ones((1,)) * (step_since_last_terminated - i),
                        infos,
                    )

            step_since_last_terminated = 0
            n_step_obs_rolling_buffer = np.zeros(
                (args.n_step,) + envs.single_observation_space.shape
            )
            n_step_action_rolling_buffer = np.zeros(
                (args.n_step,) + envs.single_action_space.shape
            )
            n_step_reward_rolling_buffer = np.zeros((args.n_step,))

            print(
                f"global_step={global_step}, episodic_return={info['episode']['r']}"
            )
            writer.add_scalar(
                "charts/episodic_return", info["episode"]["r"], global_step
            )
            writer.add_scalar(
                "charts/episodic_length", info["episode"]["l"], global_step
            )
            break

    # 5. training
    if global_step > args.learning_starts:
        if global_step % args.variable_update_period == 0:
            env_actor.load_state_dict(target_actor.state_dict())

        # 5a. calculate the target_q_values to compare with
        data = rb.sample(args.batch_size)
        with torch.no_grad():
            torch_target_mus, torch_target_stddevs = target_actor(
                data.next_observations
            )
            distribution = torch.distributions.MultivariateNormal(
                loc=torch_target_mus,
                scale_tril=torch.diag_embed(torch_target_stddevs),
            )
            torch_taus = distribution.sample(
                torch.Size([args.action_sampling_number])
            )
            completed_target_states = data.next_observations.repeat(
                [args.action_sampling_number, 1, 1]
            )
            target_qvalue_logits = target_qf(
                completed_target_states, torch_taus
            )
            next_pmfs = F.softmax(target_qvalue_logits, dim=-1).unsqueeze(-2)
            tz_hat = data.rewards + (
                args.gamma**data.bootstrapped_discounts
            ) * rd_var_support * (1 - data.dones)
            delta_z = rd_var_support[1] - rd_var_support[0]
            tz_hat_clipped = tz_hat.clamp(v_min, v_max).unsqueeze(-2)
            abs_delta_tz_hat_clipped_z = torch.abs(
                tz_hat_clipped - rd_var_support.unsqueeze(0).unsqueeze(-1)
            ).unsqueeze(0)

            target_pmfs = (
                torch.clip(1 - abs_delta_tz_hat_clipped_z / delta_z, 0, 1)
                * next_pmfs
            )
            target_pmfs = torch.sum(target_pmfs, dim=-1)
            target_pmfs = target_pmfs.mean(0)

        # 5b. define the q_value loss
        old_qval_logits = qf(data.observations, data.actions)
        old_pmfs = F.softmax(old_qval_logits, dim=1)
        old_pmfs_log = F.log_softmax(old_qval_logits, dim=1)
        old_qval = (old_pmfs * rd_var_support).sum(1)
        qvalue_loss = (-(target_pmfs * old_pmfs_log).sum(-1)).mean()

        # 5c. compute online q_values for sample actions
        stacked_observations = torch.cat(
            [data.observations, data.next_observations], dim=0
        )
        with torch.no_grad():
            target_mean, target_std = target_actor(stacked_observations)
            target_pred_distribution = torch.distributions.MultivariateNormal(
                loc=target_mean, scale_tril=torch.diag_embed(target_std)
            )
            target_pred_distribution_per_dim_constraining = (
                torch.distributions.Independent(
                    torch.distributions.Normal(
                        loc=target_mean, scale=target_std
                    ),
                    reinterpreted_batch_ndims=1,
                )
            )
            target_sampl_actions = target_pred_distribution.sample(
                torch.Size([args.action_sampling_number])
            )
            completed_states = stacked_observations.repeat(
                [args.action_sampling_number, 1, 1]
            )
            online_q_values_sampl_actions_logits = target_qf(
                completed_states, target_sampl_actions
            )
            online_q_values_sampl_actions_pmfs = torch.softmax(
                online_q_values_sampl_actions_logits, dim=-1
            )
            online_q_values_sampl_actions = (
                online_q_values_sampl_actions_pmfs * rd_var_support
            ).sum(-1)

        # 5d. calculate eta loss
        eta = F.softplus(log_eta) + _MPO_FLOAT_EPSILON
        q_logsumexp = torch.logsumexp(
            online_q_values_sampl_actions / eta, dim=0
        )
        log_num_actions = torch.log(torch.tensor(args.action_sampling_number))
        loss_eta = (
            args.epsilon_non_parametric
            + torch.mean(q_logsumexp, dim=0)
            - log_num_actions
        )
        loss_eta = eta * loss_eta

        # 5e. calculate penalty temperature
        penalty_temperature = (
            F.softplus(log_penalty_temperature) + _MPO_FLOAT_EPSILON
        )
        diff_out_of_bound = target_sampl_actions - torch.clip(
            target_sampl_actions, -1, 1
        )
        cost_out_of_bound = -torch.linalg.norm(diff_out_of_bound, dim=-1)
        penalty_impr_distr = F.softmax(
            cost_out_of_bound / penalty_temperature.detach(), dim=0
        )
        panalty_q_logsumexp = torch.logsumexp(
            cost_out_of_bound / penalty_temperature, dim=0
        )
        penalty_log_num_actions = torch.log(
            torch.tensor(args.action_sampling_number)
        )
        loss_penalty_temperature = (
            args.epsilon_penalty
            + torch.mean(panalty_q_logsumexp, dim=0)
            - penalty_log_num_actions
        )
        loss_penalty_temperature = (
            penalty_temperature * loss_penalty_temperature
        )

        # 5f. calculate improved non-parametric distribution
        impr_distr = F.softmax(
            online_q_values_sampl_actions / eta.detach(), dim=0
        )
        impr_distr += penalty_impr_distr
        loss_eta += loss_penalty_temperature

        # 5g. regression on the actions sampled by the online actor to the non-parameteric improved distribution
        alpha_mean = F.softplus(log_alpha_mean) + _MPO_FLOAT_EPSILON
        alpha_stddev = (
            torch.logaddexp(log_alpha_stddev, torch.tensor(0, device=device))
            + _MPO_FLOAT_EPSILON
        )
        online_mean, online_std = actor(stacked_observations)

        # 5h. keeping the mean from the online actor but the std from the target network
        online_pred_distribution_mean = torch.distributions.Independent(
            torch.distributions.Normal(loc=online_mean, scale=target_std),
            reinterpreted_batch_ndims=1,
        )
        online_log_probs_mean = online_pred_distribution_mean.log_prob(
            target_sampl_actions
        )  # (N,B)
        loss_policy_gradient_mean = -torch.sum(
            online_log_probs_mean * impr_distr, dim=0
        )  # (B,)
        loss_policy_gradient_mean = loss_policy_gradient_mean.mean()
        kl_mean = torch.distributions.kl.kl_divergence(
            target_pred_distribution_per_dim_constraining.base_dist,
            online_pred_distribution_mean.base_dist,
        )
        mean_kl_mean = torch.mean(kl_mean, dim=0)
        loss_kl_mean = torch.sum(alpha_mean.detach() * mean_kl_mean)
        loss_alpha_mean = torch.sum(
            alpha_mean * (args.epsilon_parametric_mu - mean_kl_mean.detach())
        )

        # 5i. keeping the mean from the target network but the std from the online actor
        online_pred_distribution_stddev = torch.distributions.Independent(
            torch.distributions.Normal(loc=target_mean, scale=online_std),
            reinterpreted_batch_ndims=1,
        )
        online_log_probs_stddev = online_pred_distribution_stddev.log_prob(
            target_sampl_actions
        )
        loss_policy_gradient_stddev = -torch.sum(
            online_log_probs_stddev * impr_distr, dim=0
        )
        loss_policy_gradient_stddev = loss_policy_gradient_stddev.mean()
        kl_stddev = torch.distributions.kl.kl_divergence(
            target_pred_distribution_per_dim_constraining.base_dist,
            online_pred_distribution_stddev.base_dist,
        )
        mean_kl_stddev = torch.mean(kl_stddev, dim=0)
        loss_kl_stddev = torch.sum(alpha_stddev.detach() * mean_kl_stddev)
        loss_alpha_stddev = torch.sum(
            alpha_stddev
            * (args.epsilon_parametric_sigma - mean_kl_stddev.detach())
        )

        # 5j. update the actor, critic networks
        actor_loss = (
            loss_policy_gradient_mean
            + loss_kl_mean
            + loss_policy_gradient_stddev
            + loss_kl_stddev
        )
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        qvalue_loss.backward()
        critic_optimizer.step()

        # 5k. update the alpha and eta tensors
        dual_loss = loss_alpha_mean + loss_alpha_stddev + loss_eta
        dual_optimizer.zero_grad()
        dual_loss.backward()
        dual_optimizer.step()

        log_eta.data.clamp_(min=_MIN_LOG_TEMPERATURE)
        log_alpha_mean.data.clamp_(min=_MIN_LOG_ALPHA)
        log_alpha_stddev.data.clamp_(min=_MIN_LOG_ALPHA)

        sgd_steps += 1

        # 5l. update the target networks
        if sgd_steps % args.target_network_update_period == 0:
            target_actor.load_state_dict(actor.state_dict())
            target_qf.load_state_dict(qf.state_dict())

        if global_step % 100 == 0:
            writer.add_scalar(
                "losses/qf_values", old_qval.mean().item(), global_step
            )
            writer.add_scalar(
                "losses/qf_loss", qvalue_loss.item(), global_step
            )
            writer.add_scalar(
                "losses/actor_loss", actor_loss.item(), global_step
            )
            writer.add_scalar(
                "losses/dual_loss", dual_loss.item(), global_step
            )

            writer.add_scalar("losses/log_eta", log_eta.item(), global_step)
            writer.add_scalar(
                "losses/log_penalty_temperature",
                log_penalty_temperature.item(),
                global_step,
            )
            writer.add_scalar(
                "losses/mean_log_alpha_mean",
                log_alpha_mean.mean().item(),
                global_step,
            )
            writer.add_scalar(
                "losses/mean_log_alpha_stddev",
                log_alpha_stddev.mean().item(),
                global_step,
            )

            writer.add_scalar(
                "losses/loss_alpha",
                (loss_alpha_mean + loss_alpha_stddev).item(),
                global_step,
            )
            writer.add_scalar("losses/loss_eta", loss_eta.item(), global_step)
            writer.add_scalar(
                "losses/kl_mean_rel",
                (mean_kl_mean / args.epsilon_parametric_mu).mean().item(),
                global_step,
            )
            writer.add_scalar(
                "losses/kl_stddev_rel",
                (mean_kl_stddev / args.epsilon_parametric_sigma).mean().item(),
                global_step,
            )

            writer.add_scalar(
                "policy/pi_stddev_min",
                online_std.min(dim=1).values.mean().item(),
                global_step,
            )
            writer.add_scalar(
                "policy/pi_stddev_max",
                online_std.max(dim=1).values.mean().item(),
                global_step,
            )
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(global_step / (time.time() - start_time)),
                global_step,
            )
