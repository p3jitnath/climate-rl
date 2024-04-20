import glob
import json
import pickle

import pandas as pd

ENV_ID = "v0-optim"
BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl"
ENV_DIR = f"{BASE_DIR}/param_tune/results/{ENV_ID}"

best_metrics = {}

for fn in glob.glob(f"{ENV_DIR}/best_*.pkl"):
    algo, kind, date = fn.split("/")[-1][:-4].split("_")[1:]

    with open(fn, "rb") as file:
        metrics = pickle.load(file)

    subset_dict = {}
    subset_dict = metrics["config"]["params"]
    subset_dict["algo"] = algo
    subset_dict["date"] = date
    subset_dict["episodic_return"] = float(metrics["last_episodic_return"])

    df = pd.DataFrame([subset_dict]).T
    df.columns = ["value"]

    print("-" * 32)
    print(df)
    print("-" * 32)

    best_metrics[algo] = subset_dict

with open(f"{ENV_DIR}/best_results.json", "w") as file:
    json.dump(best_metrics, file, ensure_ascii=False, indent=4)
