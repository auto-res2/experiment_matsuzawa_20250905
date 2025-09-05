from __future__ import annotations

"""src/main.py – experiment driver / orchestration

Run with
    python -m src.main
"""

import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .preprocess import get_data_stream
from .train import SimpleCloVeSubNet, training_loop
from .evaluate import evaluate, plot_metric

# -----------------------------------------------------------------------------
#  Project-wide paths & reproducibility helpers
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / ".research" / "iteration3" / "images"
for _d in (DATA_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED_LIST = [11, 17, 23]


# -----------------------------------------------------------------------------
#  Configuration loading
# -----------------------------------------------------------------------------

def _load_config(cfg_path: Path):
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file '{cfg_path}' not found.")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# -----------------------------------------------------------------------------
#  Main routine
# -----------------------------------------------------------------------------

def _run_experiment(exp_key: str, exp_cfg: dict):
    print(f"\n======================  Running {exp_key}  ======================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    #  Prepare continual-learning data stream
    # ------------------------------------------------------------------
    try:
        train_stream, test_stream = get_data_stream(exp_cfg["dataset"])
    except RuntimeError as e:
        # Skip experiments whose datasets are unavailable in this environment.
        print(f"[Skipped] {exp_key}: {e}\n")
        return None

    all_seeds_results: List[List[float]] = []  # → shape [n_seeds, n_tasks]

    for seed in exp_cfg.get("seeds", SEED_LIST):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SimpleCloVeSubNet(
            feat_dim=512,
            num_tasks=exp_cfg["dataset"]["num_tasks"],
            classes_per_task=exp_cfg["dataset"]["classes_per_task"],
        ).to(device)

        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=exp_cfg["optimiser"]["lr"],
            momentum=exp_cfg["optimiser"]["momentum"],
            weight_decay=exp_cfg["optimiser"]["weight_decay"],
        )

        task_acc: List[float] = []
        for task_id, (train_ds, test_ds) in enumerate(zip(train_stream, test_stream)):
            train_loader = DataLoader(
                train_ds,
                batch_size=exp_cfg["batch_size"],
                shuffle=True,
                num_workers=4,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=exp_cfg["batch_size"],
                shuffle=False,
                num_workers=4,
            )

            training_loop(
                model,
                train_loader,
                optimiser,
                device,
                task_id,
                epochs=exp_cfg["epochs_per_task"],
            )

            acc = evaluate(model, test_loader, task_id, device)
            task_acc.append(acc)
            print(f"Seed {seed} – Task {task_id:02d} | Accuracy: {acc:.2%}")

        all_seeds_results.append(task_acc)

    # ------------------------------------------------------------------
    #  Aggregate over seeds and save / plot results
    # ------------------------------------------------------------------
    arr = np.array(all_seeds_results)  # shape = [seeds, tasks]
    acc_mean = arr.mean(axis=0)
    acc_std = arr.std(axis=0)

    df = pd.DataFrame({
        "task": np.arange(len(acc_mean)),
        "acc_mean": acc_mean,
        "acc_std": acc_std,
    })

    print("\n=====  Numerical Results (mean ± std across seeds)  =====")
    print(df.to_string(index=False, float_format="{:.4f}".format))

    fig_name = f"accuracy_{exp_key}.pdf"
    plot_metric(
        x=df["task"].tolist(),
        y=df["acc_mean"].tolist(),
        ylabel="Accuracy",
        title=f"{exp_key.upper()} – Accuracy over Tasks",
        fname=fig_name,
    )

    return df


def main():
    """Entry point when running ``python -m src.main``."""

    cfg = _load_config(ROOT / "config" / "config.yaml")
    exps = cfg.get("experiments", {})

    results = {}
    for key, exp_cfg in exps.items():
        df = _run_experiment(key, exp_cfg)
        if df is not None:
            results[key] = df

    # Optionally save aggregated results for downstream consumption
    if results:
        out_path = ROOT / ".research" / "iteration3" / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {k: v.to_dict(orient="list") for k, v in results.items()}
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)
        print(f"\n[Results saved] {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
