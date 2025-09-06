"""src/main.py – orchestrates the complete experimental workflow"""
from __future__ import annotations

# std -----------------------------------------------------------------------
import json, os, random, sys, pathlib
from typing import Any

# third-party ---------------------------------------------------------------
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

# local modules --------------------------------------------------------------
from .train import (
    build_backbone,
    ExpandingClassifier,
    LOSRMemory,
    ERBuffer,
    train_stream,
)
from .evaluate import evaluate, plot_curves
from .preprocess import build_split_cifar100

# ============================================================================
# Simple utilities -----------------------------------------------------------

Path = pathlib.Path
IMAGES_DIR = Path(".research/iteration6/images")
RESULTS_DIR = Path(".research/iteration6/results")


def gpu_assert() -> None:
    if not torch.cuda.is_available():
        sys.exit("[ERROR] GPU runner required – aborting as per spec.")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str | Path) -> Any:  # type: ignore[override]
    with open(path) as fp:
        return yaml.safe_load(fp)


def dump_json(obj: Any, path: Path) -> None:  # noqa: D401
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2)
    print(json.dumps(obj, indent=2))

# ============================================================================
# Main experiment logic ------------------------------------------------------

gpu_assert()
CONFIG = load_yaml("config/config.yaml")
shared = CONFIG["shared"]


def run_one(exp_cfg: dict, seed: int) -> None:
    seed_all(seed)
    device = "cuda"

    # ------------------------------- data
    train_stream_ds, testset = build_split_cifar100(seed)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    # ------------------------------- model
    backbone = build_backbone(exp_cfg["backbone"]).to(device)
    clf = ExpandingClassifier(256).to(device)

    strategy = exp_cfg["method"]
    losr = None
    buffer = None
    params = list(backbone.parameters()) + list(clf.parameters())
    if strategy == "LOSR":
        losr = LOSRMemory(256, budget_kb=shared["budget_kb"]).to(device)
        params += list(losr.synth.parameters())
    elif strategy == "ER":
        buffer = ERBuffer(shared["budget_kb"] * 1024)

    optimiser = torch.optim.AdamW(params, lr=shared["lr_grid"][0], weight_decay=1e-4)

    # ------------------------------- continual training
    for subset in train_stream_ds:
        loader = DataLoader(
            subset, batch_size=shared["batch_size"], shuffle=True, num_workers=4
        )
        for _ in range(shared["passes_per_task"]):
            train_stream(backbone, clf, strategy, loader, optimiser, losr, buffer, device)

    # ------------------------------- evaluation
    acc = evaluate(backbone, clf, test_loader, device)
    mem_kb = 0.0
    if losr is not None:
        mem_kb = losr.bytes() / 1024
    elif buffer is not None:
        mem_kb = buffer.bytes() / 1024

    result = {
        "method": strategy,
        "seed": seed,
        "accuracy": acc,
        "memory_kB": mem_kb,
    }
    out_json = RESULTS_DIR / f"{strategy}_seed{seed}.json"
    dump_json(result, out_json)

    # quick sanity plot
    plot_curves([0, 1], [0, acc * 100], "", "Accuracy %", strategy, IMAGES_DIR / f"acc_{strategy}.pdf")


# ---------------------------------------------------------------------------

def main() -> None:
    for exp in CONFIG["experiments"]:
        for s in shared["seeds"]:
            run_one(exp, s)


if __name__ == "__main__":
    main()
