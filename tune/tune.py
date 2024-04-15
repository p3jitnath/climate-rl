import json
import os
import pickle
import random
import subprocess
import time
from dataclasses import dataclass

import ray
import tyro
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

from tune.config import config

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl"


@dataclass
class Args:
    algo: str = "td3"
    """name of rl-algo"""


def objective(config):
    optuna_id = random.randint(1000000000, 9999999999)
    algo = "reinforce"
    tmp_file = f"{algo}_{optuna_id}.tmp"
    results_path = f"{BASE_DIR}/tune/tmp/{tmp_file}"

    cmd = f"""python {BASE_DIR}/rl-algos/{algo}/main.py --optimise --write-to-file {results_path} """
    for param in config["params"]:
        cmd += f"""--{param} {config['params'][{param}]} """

    subprocess.run(cmd.split())

    while not os.path.exists(results_path):
        time.sleep(1)

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
if os.environ["ip_head"]:
    ray_kwargs["address"] = os.environ["ip_head"]
ray.init(**ray_kwargs)

trainable_with_resources = tune.with_resources(
    objective, resources={"cpu": 2, "gpu": 1}
)
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="last_episodic_return",
        mode="max",
        search_alg=search_alg,
        num_samples=2,
        max_concurrent_trials=2,
    ),
    param_space={
        "scaling_config": train.ScalingConfig(use_gpu=True),
        "params": search_space,
    },
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)

with open(
    f"{BASE_DIR}/tune/results/best_{args.algo}_config_{date}.json", "w"
) as file:
    json.dump(results, file, ensure_ascii=False, indent=4)
