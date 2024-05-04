import json
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import ray
import tyro
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl"
sys.path.append(BASE_DIR)

from param_tune.config import config


@dataclass
class Args:
    algo: str = "ddpg"
    """name of rl-algo"""
    exp_id: str = "v0-optim-L-60k"
    """name of the experiment"""
    env_id: str = "SimpleClimateBiasCorrection-v0"
    """name of the environment"""
    actor_layer_size: Optional[int] = None
    """layer size for the actor network"""
    critic_layer_size: Optional[int] = None
    """layer size for the critic network"""
    opt_timesteps: int = 2000
    """timestep duration for one single optimisation run"""


def objective(config):
    study_id = random.randint(1000000000, 9999999999)
    tmp_file = f"{args.algo}_{study_id}.tmp"
    results_path = f"{BASE_DIR}/param_tune/tmp/{tmp_file}"

    cmd = f"""python -u {BASE_DIR}/rl-algos/{args.algo}/main.py --env_id {args.env_id} --optimise --opt_timesteps {args.opt_timesteps} --write-to-file {results_path} """
    for param in config["params"]:
        if param == "actor_critic_layer_size":
            actor_layer_size = (
                args.actor_layer_size
                if args.actor_layer_size
                else config["params"][param]
            )
            critic_layer_size = (
                args.critic_layer_size
                if args.critic_layer_size
                else config["params"][param]
            )
            cmd += f"""--actor_layer_size {actor_layer_size} """
            cmd += f"""--critic_layer_size {critic_layer_size} """
        else:
            cmd += f"""--{param} {config['params'][param]} """

    subprocess.run(cmd.split())

    counter = 0
    while not os.path.exists(results_path):
        time.sleep(10)
        counter += 1
        if counter >= 12:
            raise RuntimeError("An error has occured.")

    with open(results_path, "rb") as f:
        results_dict = pickle.load(f)
        train.report(
            {"last_episodic_return": results_dict["last_episodic_return"]}
        )


args = tyro.cli(Args)
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))

search_space = config[args.algo]
search_alg = OptunaSearch(seed=42)

ray_kwargs = {}
ray_kwargs["runtime_env"] = {"working_dir": BASE_DIR, "conda": "venv"}
try:
    if os.environ["ip_head"]:
        ray_kwargs["address"] = os.environ["ip_head"]
except Exception:
    ray_kwargs["num_cpus"] = 2

ray.init(**ray_kwargs)

trainable = tune.with_resources(objective, resources={"cpu": 1, "gpu": 0.25})

RESULTS_DIR = f"{BASE_DIR}/param_tune/results/{args.exp_id}"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

storage_path = f"{RESULTS_DIR}/{args.algo}_run_{date}"

if tune.Tuner.can_restore(storage_path):
    print("Restoring old run ...")
    tuner = tune.Tuner.restore(
        storage_path, trainable=trainable, resume_errored=True
    )
else:
    print("Starting from scratch ...")
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="last_episodic_return",
            mode="max",
            search_alg=search_alg,
            num_samples=100,
            max_concurrent_trials=32,
        ),
        param_space={
            "scaling_config": train.ScalingConfig(use_gpu=True),
            "params": search_space,
        },
        run_config=train.RunConfig(
            storage_path=storage_path,
            name=f"pn341_ray_slurm_{args.exp_id}_{args.algo}",
        ),
    )
results = tuner.fit()
best_result = results.get_best_result()
print("Best metrics:", best_result.metrics)

with open(f"{RESULTS_DIR}/best_{args.algo}_result_{date}.pkl", "wb") as file:
    pickle.dump(best_result.metrics, file)
