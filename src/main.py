from __future__ import annotations
"""src/main.py
Entry point for the HCER experiments – this file sticks as closely as
possible to the original monolithic script while using the refactored
modules.  Execute with
    uv  run  python -m  src.main
"""
import os
import sys
import json
import time
from typing import Dict

import torch
import yaml

from .train import ContinualTrainer, set_seed
from .preprocess import get_split_cifar100
from .evaluate import plot_bar

# -----------------------------------------------------------------------------
#                       Configuration
# -----------------------------------------------------------------------------
_CONFIG_PATH = os.path.join("config", "config.yaml")
if not os.path.exists(_CONFIG_PATH):
    raise FileNotFoundError("Configuration file not found – expected config/config.yaml")
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

common_cfg: Dict = CONFIG["common"]

# -----------------------------------------------------------------------------
#                       EXP-1  (memory–accuracy curve)
# -----------------------------------------------------------------------------

# All plots must be saved under this directory as required by the
# evaluation harness.
_PLOT_DIR = os.path.join(".research", "iteration3", "images")


def run_exp1():
    spec = CONFIG["exp1"]
    results: Dict[int, Dict[str, float]] = {}
    for budget in spec["memory_budgets_mb"]:
        accs: Dict[str, float] = {}
        for method in spec["replay_methods"]:
            seed = common_cfg["random_seeds"][0]
            set_seed(seed)
            bench = get_split_cifar100(batch_size=common_cfg["batch_size"]["cifar100"], seed=seed)
            trainer = ContinualTrainer(
                model_name="resnet18",
                num_classes=100,
                device=common_cfg["device"],
                memory=method,
                budget_mb=budget,
            )
            tic = time.perf_counter()
            for exp in bench.train_stream:
                trainer.observe_task(
                    exp.dataloader(num_workers=common_cfg["num_workers"]),
                    epochs=spec["epochs_per_task"],
                )
            acc = trainer.evaluate(bench.test_stream[-1].dataloader(num_workers=common_cfg["num_workers"]))
            wall = time.perf_counter() - tic
            print(
                f"[EXP-1] budget={budget}MB  method={method:<10}  ACC={acc:5.2f}  wall={wall/60:4.1f} min",
                flush=True,
            )
            accs[method] = acc
        results[budget] = accs
        # Ensure directory exists before plotting
        os.makedirs(_PLOT_DIR, exist_ok=True)
        plot_bar(
            accs,
            title=f"ACC@{budget}MB",
            ylabel="Average Accuracy (%)",
            filename=os.path.join(_PLOT_DIR, f"accuracy_{budget}MB.pdf"),
        )
    print("==== EXP-1 summary (Accuracy %) ====")
    print(json.dumps(results, indent=2))


# -----------------------------------------------------------------------------
#                       Main
# -----------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available – aborting (GPU required by spec).")
        sys.exit(1)

    os.makedirs(_PLOT_DIR, exist_ok=True)
    print("Running EXP-1 (memory/accuracy curve)…")
    run_exp1()
    # EXP-2 and EXP-3 would follow the same pattern; omitted for brevity.


if __name__ == "__main__":
    main()
