from __future__ import annotations

"""src/main.py
-------------------------------------------------------------------------------
Entry-point.  Usage: ``python -m src.main``

The script orchestrates a single representative continual-learning experiment
for CI / demo purposes.  Results (JSON) are stored under
``.research/iteration4`` and figures under ``.research/iteration4/images`` as
mandated by the prompt.
"""

import json
from pathlib import Path
from typing import List

import torch
import yaml
from torch.utils.data import DataLoader

from .preprocess import (
    RESULTS_DIR,
    IMAGES_DIR,
    build_cifar100_benchmark,
    build_mini_imgnet_benchmark,
    seed_everything,
)
from .train import (
    LOSRMemory,
    ExpandingClassifier,
    build_backbone,
    train_one_experience,
)
from .evaluate import accuracy, barplot_accuracy, save_json

CONFIG_PATH = Path("config") / "config.yaml"


def _load_cfg() -> List[dict]:
    with CONFIG_PATH.open() as fp:
        cfg = yaml.safe_load(fp)
    return cfg["experiments"]


def _ensure_classifier_capacity(clf: ExpandingClassifier, labels: torch.Tensor):
    """Grow classifier so that ``clf.out_dim > labels.max()``."""
    needed = int(labels.max().item()) + 1
    if needed > clf.out_dim:
        clf.add_classes(needed - clf.out_dim)


def _run_single(exp_cfg: dict):
    print("\n=== Running experiment ===")
    print(json.dumps(exp_cfg, indent=2))

    seed_everything(exp_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------- data -----------------------------------------------------
    if exp_cfg["dataset"] == "cifar100":
        bench = build_cifar100_benchmark(seed=exp_cfg["seed"])
        train_stream = bench.train_stream
        test_loader = DataLoader(bench.test_stream[0].dataset, batch_size=256, shuffle=False, num_workers=4)
    elif exp_cfg["dataset"] == "mini_imagenet":
        bench = build_mini_imgnet_benchmark(seed=exp_cfg["seed"])
        train_stream = bench["train_stream"]
        test_loader = DataLoader(bench["test_set"], batch_size=256, shuffle=False, num_workers=4)
    else:
        raise ValueError(exp_cfg["dataset"])

    # --------------- model ----------------------------------------------------
    backbone = build_backbone(exp_cfg["backbone"]).to(device)
    clf = ExpandingClassifier().to(device)

    losr = None
    if exp_cfg["method"] == "LOSR":
        losr = LOSRMemory(feat_dim=256, r=2, budget_kb=exp_cfg["budget_kb"]).to(device)
        params = list(backbone.parameters()) + list(clf.parameters()) + list(losr.parameters())
    else:
        params = list(backbone.parameters()) + list(clf.parameters())

    optimiser = torch.optim.AdamW(params, lr=exp_cfg["lr"], weight_decay=1e-4)

    # --------------- training loop over 20 experiences -----------------------
    for exp_ds in train_stream:
        # exp_ds is a ``Subset`` – pass it directly to retain index subset
        loader = DataLoader(exp_ds, batch_size=128, shuffle=True, num_workers=4)
        train_one_experience(backbone, clf, losr, loader, optimiser, device)
        # simple cosine decay proxy
        for pg in optimiser.param_groups:
            pg["lr"] *= 0.95

    # --------------- evaluation ----------------------------------------------
    acc = accuracy(backbone, clf, test_loader, device)

    result = {
        "dataset": exp_cfg["dataset"],
        "backbone": exp_cfg["backbone"],
        "method": exp_cfg["method"],
        "mem_kB": losr.bytes() / 1024 if losr else exp_cfg["budget_kb"],
        "final_accuracy": acc,
        "seed": exp_cfg["seed"],
    }

    # Persist & display JSON ---------------------------------------------------
    out_json = RESULTS_DIR / f"exp1_{exp_cfg['dataset']}_{exp_cfg['backbone']}_{exp_cfg['method']}_{exp_cfg['seed']}.json"
    save_json(result, out_json)

    # quick bar plot -----------------------------------------------------------
    barplot_accuracy(exp_cfg["dataset"], exp_cfg["backbone"], exp_cfg["method"], acc, IMAGES_DIR)


# -----------------------------------------------------------------------------
#  Main – run only a *single* representative experiment for CI / demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfgs = _load_cfg()

    # Keep runtime small:  run just the first config in the list --------------
    _run_single(cfgs[0])
