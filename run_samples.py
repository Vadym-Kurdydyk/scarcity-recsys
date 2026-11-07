"""Generates N samples of a dataset, computes the EVs of every sample and runs the
recommenders on it with Elliot. Results are appended to ev_results/<dataset>.csv
(one row per sample: EVs + nDCG@100 / MAP@100 / nDCG@10 of every model), so the
script can be stopped and restarted and it continues where it stopped.

usage: python run_samples.py <dataset> <n_samples>
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from ev_utils import compute_evs, load_ratings, sample_urm

PYTHON = sys.executable
CUTOFFS = [10, 100]
METRICS = ["nDCG", "MAP"]
EPOCHS = 20

# same models as in the paper as far as Elliot has them (see notebook for the mapping)
MODELS = f"""
    UserKNN:
      meta:
        save_recs: False
      neighbors: 20
      similarity: [cosine, pearson, asymmetric, jaccard]
      implementation: aiolli
    ItemKNN:
      meta:
        save_recs: False
      neighbors: 20
      similarity: [cosine, pearson, adjusted]
      implementation: aiolli
    MF:
      meta:
        save_recs: False
        validation_rate: {EPOCHS}
      epochs: {EPOCHS}
      batch_size: 512
      factors: 10
      lr: 0.01
      reg: 0.02
    FunkSVD:
      meta:
        save_recs: False
        validation_rate: {EPOCHS}
      epochs: {EPOCHS}
      batch_size: 512
      factors: 10
      lr: 0.01
      reg_w: 0.02
      reg_b: 0.02
    BPRMF:
      meta:
        save_recs: False
        validation_rate: {EPOCHS}
      epochs: {EPOCHS}
      factors: 10
      lr: 0.05
    NonNegMF:
      meta:
        save_recs: False
        validation_rate: {EPOCHS}
      epochs: {EPOCHS}
      batch_size: 512
      factors: 15
      lr: 0.01
      reg: 0.02
    PMF:
      meta:
        save_recs: False
        validation_rate: {EPOCHS}
      epochs: {EPOCHS}
      batch_size: 512
      factors: 10
      lr: 0.01
      reg: 0.001
      gaussian_variance: 0.1
"""

CONFIG = """experiment:
  version: 0.3.1
  dataset: {name}
  data_config:
    strategy: dataset
    dataset_path: {folder}/ratings.tsv
  splitting:
    test_splitting:
        strategy: random_subsampling
        test_ratio: 0.2
  top_k: 100
  evaluation:
    cutoffs: {cutoffs}
    simple_metrics: {metrics}
    relevance_threshold: 1
  gpu: -1
  path_output_rec_result: {folder}/recs/
  path_output_rec_weight: {folder}/weights/
  path_output_rec_performance: {folder}/performance/
  path_log_folder: {folder}/log/
  models:
""" + MODELS

SIM_NAMES = {"UserKNN": ["cosine", "pearson", "asymmetric", "jaccard"], "ItemKNN": ["cosine", "pearson", "adjusted"]}


def read_performance(folder):
    """Elliot writes one tsv per cutoff with the best config of each model and one
    tsv per grid model (UserKNN / ItemKNN) with every similarity."""
    out = {}
    for k in CUTOFFS:
        best = pd.read_csv(sorted(glob.glob(f"{folder}/performance/rec_cutoff_{k}_*.tsv"))[-1], sep="\t")
        for _, row in best.iterrows():
            model = row["model"].split("_")[0]
            if model in SIM_NAMES:
                continue
            for m in METRICS:
                out[f"{model}_{m}@{k}"] = row[m]
        for knn in SIM_NAMES:
            grid = pd.read_csv(sorted(glob.glob(f"{folder}/performance/rec_{knn}_cutoff_{k}_*.tsv"))[-1], sep="\t")
            for _, row in grid.iterrows():
                sim = row["model"].split("sim=")[1].split("_")[0]
                for m in METRICS:
                    out[f"{knn}-{sim}_{m}@{k}"] = row[m]
    return out


def main(name, n_samples):
    df = load_ratings(f"data/{name}/{name}_ratings.tsv")
    os.makedirs("ev_results", exist_ok=True)
    out_file = f"ev_results/{name}.csv"
    done = pd.read_csv(out_file) if os.path.exists(out_file) else pd.DataFrame(columns=["sample"])

    for s in range(n_samples):
        if s in set(done["sample"]):
            continue
        start = time.time()
        rng = np.random.default_rng(1000 + s)
        sample = sample_urm(df, rng)
        if sample is None:
            print(f"[{name}] sample {s}: no valid sample found")
            continue

        folder = os.path.abspath(f"data/samples/{name}/s{s}").replace("\\", "/")
        os.makedirs(folder, exist_ok=True)
        sample.to_csv(f"{folder}/ratings.tsv", sep="\t", header=False, index=False)
        with open(f"{folder}/config.yml", "w") as f:
            f.write(CONFIG.format(name=f"{name}_s{s}", folder=folder, cutoffs=CUTOFFS, metrics=METRICS))

        with open(f"{folder}/elliot_output.txt", "w") as log:
            p = subprocess.run([PYTHON, "-c", f"from elliot.run import run_experiment; run_experiment(r'{folder}/config.yml')"],
                               stdout=log, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(f"[{name}] sample {s}: elliot failed, see {folder}/elliot_output.txt")
            continue

        row = {"sample": s, "n_users": sample["user"].nunique(), "n_items": sample["item"].nunique(),
               "n_ratings": len(sample)}
        row.update(compute_evs(sample))
        row.update(read_performance(folder))
        done = done.append(row, ignore_index=True)
        done.to_csv(out_file, index=False)
        print(f"[{name}] sample {s}: {row['n_users']} users, {row['n_items']} items, {row['n_ratings']} ratings, "
              f"{time.time() - start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
