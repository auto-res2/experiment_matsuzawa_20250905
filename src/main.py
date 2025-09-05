"""src/main.py
Main orchestration script.  Run with:

    python -m src.main

No command-line arguments are needed; everything is controlled via
`config/config.yaml`.  The script performs the following steps:
  1.  Load YAML configuration
  2.  Prepare deterministic RNG & torch back-end
  3.  Run the Stage-0 sanity gate
  4.  Launch training for the list of seeds
  5.  Save numerical results and generate accuracy figure
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .evaluate import MetricLogger, plot_accuracy, stage0_sanity
from .preprocess import build_stream
from .train import CLoVeSub

# ----------------------------------------------------------------------
#  0.  Load configuration ------------------------------------------------
# ----------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    print(f"[FATAL] Configuration file not found at {CONFIG_PATH}", file=sys.stderr)
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# ----------------------------------------------------------------------
#  1.  Deterministic setup ----------------------------------------------
# ----------------------------------------------------------------------
SEED_BASE = CONFIG["global"].get("seeds", [11])[0]
random.seed(SEED_BASE)
np.random.seed(SEED_BASE)
torch.manual_seed(SEED_BASE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_BASE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------
#  2.  Stage-0 sanity gate ----------------------------------------------
# ----------------------------------------------------------------------
print("\n[INFO] Building model for Stage-0 sanity checks …")
model_tmp = CLoVeSub(CONFIG)
stage0_sanity(model_tmp, CONFIG)
del model_tmp  # free VRAM before real training

# ----------------------------------------------------------------------
#  3.  Data pipeline -----------------------------------------------------
# ----------------------------------------------------------------------
print("[INFO] Building continual data stream …")
train_stream, val_stream, test_stream = build_stream(CONFIG)

# ----------------------------------------------------------------------
#  4.  Training loop for each seed --------------------------------------
# ----------------------------------------------------------------------
RESULT_DIR = Path("results")
FIG_DIR = Path("figures")
RESULT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

results = {s: [] for s in CONFIG["global"]["seeds"]}

for seed in CONFIG["global"]["seeds"]:
    print(f"\n[RUN] Starting training with seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = CLoVeSub(CONFIG)
    logger = MetricLogger(RESULT_DIR / f"seed{seed}.json")

    for task_id, (tr_loader, va_loader, te_loader) in enumerate(
        zip(train_stream, val_stream, test_stream)
    ):
        print(f"[Seed {seed}] Training task {task_id} …")
        t0 = time.time()
        model.train_task(tr_loader, va_loader, task_id, logger)
        dt = time.time() - t0
        acc = model.evaluate_task(te_loader)
        results[seed].append(acc)
        logger.log(f"task{task_id}/test_acc", acc)
        logger.log(f"task{task_id}/train_time_s", dt)
    logger.close()

# ----------------------------------------------------------------------
#  5.  Summary & visualisation ------------------------------------------
# ----------------------------------------------------------------------
import pandas as pd  # pylint: disable=wrong-import-position

summary = pd.DataFrame({s: results[s] for s in results}).mean(axis=1)
print("\nExperiment description: CIFAR-100 split 20×5 – CLoVe-Sub <1 MB buffer")
print("Average accuracy per task:")
print(summary.values)

plot_accuracy(summary.tolist(), FIG_DIR / "accuracy.pdf")
