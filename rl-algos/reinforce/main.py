import json
import os
import random
import time
from dataclasses import dataclass

import climate_envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from reinforce_actor import Actor
from torch.utils.tensorboard import SummaryWriter

from tune.utils.no_op_summary_writer import NoOpSummaryWriter

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl"

with open(f"{BASE_DIR}/rl-algos/config.json", "r") as file:
    config = json.load(file)

os.environ["MUJOCO_GL"] = "egl"
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))


@dataclass
class Args:
    exp_name: str = "reinforce_torch"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "personal-p3jitnath"
    """the entity (team) of wandb's project"""
    wandb_group: str = date
    """the group name under wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = config["env_id"]
    """the environment id of the environment"""
    total_timesteps: int = config["total_timesteps"]
    """total timesteps of the experiments"""
    num_steps: int = config["max_episode_steps"]
    """the number of steps to run in each environment per policy rollout"""
    num_envs: int = 1
    """the number of sequential game environments"""

    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""

    optimise: bool = False
    """whether to modify output for hyperparameter optimisation"""
    write_to_file: str = ""
    """filename to write last episode return"""

    def __post_init__(self):
        if self.optimise:
            self.track = False
            self.capture_video = False
            self.total_timesteps = config["opt_timesteps"]


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: x % 100 == 0,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


args = tyro.cli(Args)
run_name = f"{args.wandb_group}/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        group=args.wandb_group,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )


if args.optimise:
    writer = NoOpSummaryWriter()
else:
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
print(f"device: {device}")

# 0. env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
)

assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"

actor = Actor(envs).to(device)
optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)
envs.single_observation_space.dtype = np.float32


# util function to calculate the discounted normalized returns
def compute_returns(rewards):
    rewards = np.array(rewards)
    returns = np.zeros_like(rewards, dtype=np.float32)
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + args.gamma * R
        returns[t] = R
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    return returns


# 1. start the game
global_step = 0
start_time = time.time()
obs, _ = envs.reset(seed=args.seed)
args.num_episodes = args.total_timesteps // args.num_steps

for episode in range(args.num_episodes):
    log_probs, rewards = [], []
    for step in range(args.num_steps):
        # 2. retrieve action(s)
        global_step += args.num_envs
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        action, log_prob, _ = actor.get_action(obs)

        # 3. execute the game and log data
        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )

        if "final_info" in infos:
            for info in infos["final_info"]:
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

        # 4. save data to the list of records
        log_probs.append(log_prob)
        rewards.append(reward)
        obs = next_obs

    # 5. compute the returns and the policy loss
    returns = compute_returns(rewards)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    actor_loss = -(torch.stack(log_probs) * returns).sum()

    optimizer.zero_grad()
    actor_loss.backward()
    optimizer.step()

    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
    writer.add_scalar(
        "charts/SPS",
        int(global_step / (time.time() - start_time)),
        global_step,
    )

    if episode == args.num_episodes - 1:
        if args.write_to_file:
            episodic_return = info["episode"]["r"][0]
            with open(args.write_to_file, "wb") as file:
                import pickle

                pickle.dump(
                    {
                        "num_episodes": args.num_episodes,
                        "last_episodic_return": episodic_return,
                    },
                    file,
                )

envs.close()
writer.close()
