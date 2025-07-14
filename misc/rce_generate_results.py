import multiprocessing
import os
import re
from dataclasses import dataclass
from functools import partial
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tyro

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings

import tensorflow as tf
from tensorflow.core.util import event_pb2

warnings.filterwarnings("ignore")

BASE_DIR = "/gws/nopw/j04/ai4er/users/pn341/climate-rl"
RUNS_DIR = f"{BASE_DIR}/runs"
TABLES_DIR = f"{BASE_DIR}/results/tables/"
IMGS_DIR = f"{BASE_DIR}/results/imgs/"

THRESHOLDS = {"rce-v0": 4.39e4, "rce17-v0": 4.37e4, "rce17-v1": 4.365e4}


@dataclass
class Args:
    exp_id: str = ""
    """the id of this experiment"""


args = tyro.cli(Args)

EXP_ID = args.exp_id


def extract_version(text):
    pattern = r"(?:\b\w+-)?v\d+"
    match = re.search(pattern, text)
    return match.group(0) if match else None


THRESHOLD = THRESHOLDS[extract_version(EXP_ID)]
THRESHOLD += 0.005e4 if "homo" in EXP_ID else 0
THRESHOLD_EPISODE = 10

EPISODE_COUNT = 10000 // 500
N_EXPERIMENTS = 10
TOP_K = 3

algos = ["ddpg", "dpg", "ppo", "reinforce", "sac", "td3", "trpo", "tqc", "avg"]


def retrieve_data(algo, ep10k):
    exp_id = EXP_ID
    if ep10k and "10k" not in EXP_ID:
        exp_id = exp_id + "-10k"
    tfrecord_paths = glob(f"{RUNS_DIR}/{exp_id}_*/*_{algo}_*/*")
    tfrecord_paths += glob(f"{RUNS_DIR}/x9_{exp_id}_*/*_{algo}_*/*")
    cntr = 0
    data = {}
    for tfrecord_path in tfrecord_paths:
        seed = int(tfrecord_path.split("/")[-2].split("__")[-2])
        data[seed] = []
        episodic_idx = 0
        serialized_events = tf.data.TFRecordDataset(tfrecord_path)
        for serialized_example in serialized_events:
            e = event_pb2.Event.FromString(serialized_example.numpy())
            for v in e.summary.value:
                if (
                    v.HasField("simple_value")
                    and v.tag == "charts/episodic_return"
                ):
                    episodic_idx += 1
                    data[seed].append(
                        {
                            "episode": episodic_idx,
                            "episodic_return": -v.simple_value,  # Negate here to simplify later processing
                        }
                    )
        cntr += 1
    assert cntr == N_EXPERIMENTS, f"NOT OK ... {algo} ... found {cntr}"
    return data


def parallel_retrieve_data(algo, ep10k):
    return algo, retrieve_data(algo, ep10k)


def retrieve_plot_data(data, algo):
    episode_data = {}
    for seed_records in data[algo].values():
        for record in seed_records:
            episode = record["episode"]
            if episode not in episode_data:
                episode_data[episode] = []
            episode_data[episode].append(record["episodic_return"])

    episodes = sorted(episode_data.keys())
    returns = [episode_data[ep] for ep in episodes]

    means = [np.mean(x) for x in returns]
    std_devs = [np.std(x) / np.sqrt(N_EXPERIMENTS) for x in returns]

    assert (
        len(means) == len(std_devs) == EPISODE_COUNT
    ), f"EPISODES NOT OK ... {algo}"
    return episodes, returns, means, std_devs


with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    results = pool.map(partial(parallel_retrieve_data, ep10k=False), algos)
    results_10k = pool.map(partial(parallel_retrieve_data, ep10k=True), algos)

data = {algo: result for algo, result in results}
data_10k = {algo: result for algo, result in results_10k}

data_v2 = {}
data_v2_10k = {}

for algo in algos:
    _, _, means, std_devs = retrieve_plot_data(data, algo)
    _, _, means_10k, std_devs_10k = retrieve_plot_data(data_10k, algo)
    data_v2[algo] = {"means": means, "std_devs": std_devs}
    data_v2_10k[algo] = {"means": means_10k, "std_devs": std_devs_10k}


def calculate_score(data, threshold, alpha=0.9, beta=0.05, gamma=0.05):
    scores = []
    # best_algo = min(
    #     data_v2_10k.keys(), key=lambda algo: data_v2_10k[algo]["means"][-1]
    # )
    for algo, performance in data.items():
        episodes_to_threshold = next(
            (
                idx
                for idx, val in enumerate(performance["means"])
                if val <= threshold
            ),
            None,
        )
        if episodes_to_threshold is not None:
            var_after_threshold = np.mean(
                np.power(performance["std_devs"][episodes_to_threshold:], 2)
            )
            mean_at_threshold = performance["means"][episodes_to_threshold]
            diff_from_10k = -1 * (
                data_v2_10k[algo]["means"][-1] - mean_at_threshold
            )
        else:
            var_after_threshold = np.NaN
            mean_at_threshold = np.NaN
            diff_from_10k = np.NaN
        scores.append(
            {
                "algo": algo,
                "steps_to_threshold": (
                    500 * episodes_to_threshold
                    if episodes_to_threshold is not None
                    else np.NaN
                ),
                "mean_at_threshold": -1 * mean_at_threshold,
                "var_after_threshold": var_after_threshold,
                "diff_from_10k@10k": diff_from_10k,
            }
        )
    return scores


import pandas as pd

df = pd.DataFrame(calculate_score(data_v2, threshold=THRESHOLD))
df = df.set_index("algo")

df["abs_diff_from_10k@10k"] = df["diff_from_10k@10k"].abs()

rank_metrics = [
    "steps_to_threshold",
    "var_after_threshold",
    "abs_diff_from_10k@10k",
]
ranked = df[rank_metrics].rank(ascending=True, na_option="bottom")

penalty = 10  # Fixed penalty added to the rank
mask = df["var_after_threshold"] > 3e5
ranked.loc[mask, "var_after_threshold"] += penalty
mask = df["steps_to_threshold"] > THRESHOLD_EPISODE * 500
ranked.loc[mask, "var_after_threshold"] += penalty

df["final_score"] = ranked.sum(axis=1)
df.loc[df[rank_metrics].isnull().any(axis=1), "final_score"] = float("nan")
df = df.sort_values("final_score")

# df

df.to_csv(TABLES_DIR + f"{EXP_ID}.csv")


def format_func(value, tick_number=None):
    num_thousands = (
        0 if np.abs(value) < 1000 else int(np.floor(np.log10(abs(value)) / 3))
    )
    value = round(value / 1000**num_thousands, 2)
    txt = f"{value:g}" + " KMGTPEZY"[num_thousands]
    return txt


plt.rcParams.update({"font.size": 12})

selected_algos = df.index.values[:TOP_K]

fig, ax = plt.subplots(figsize=(6.4, 4.8))

for idx, algo in enumerate(selected_algos):
    episodes, _, means, std_devs = retrieve_plot_data(data, algo)
    global_steps = [500 * x for x in episodes]
    (line,) = plt.plot(global_steps, means, label=f"{idx+1}. {algo.upper()}")
    plt.fill_between(
        global_steps,
        np.array(means) - (1.96 * np.array(std_devs)),  # 95% CI
        np.array(means) + (1.96 * np.array(std_devs)),  # 95% CI
        color=line.get_color(),
        alpha=0.3,
    )

plt.axhline(THRESHOLD, c="k", ls="--", label="THRESH")
plt.title(f"{EXP_ID}")
plt.xlabel("# Steps")
plt.ylabel("$-\mathrm{(Episodic~Return)}$")
plt.grid(True, which="both", ls="--")
plt.yscale("log")
plt.legend()
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

plt.savefig(
    f"{IMGS_DIR}/{EXP_ID}_log10_top{TOP_K}_episodic_returns.png",
    dpi=300,
    bbox_inches="tight",
)
# plt.show()
