import os
import pickle
import random
import subprocess
import time

import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl"


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


search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-2),
    "gamma": tune.uniform(0.89, 0.99),
}
search_alg = OptunaSearch(seed=42)

ray.init(
    runtime_env={
        "working_dir": BASE_DIR,
        "conda": "venv",
    },
)
trainable_with_resources = tune.with_resources(objective, resources={"gpu": 1})
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="last_episodic_return",
        mode="max",
        search_alg=search_alg,
        num_samples=10,
        max_concurrent_trials=2,
    ),
    param_space={
        "scaling_config": train.ScalingConfig(use_gpu=True),
        "params": search_space,
    },
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
