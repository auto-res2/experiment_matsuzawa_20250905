from __future__ import annotations

"""
main.py – orchestrates the whole experimental pipeline
Entry-point:  python main.py
"""

import json
import pathlib
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from evaluate import evaluate, plot_line
from preprocess import build_cifar100_one_class, build_split_cifar100
from train import (
    ERBuffer,
    ExpandingClassifier,
    LOSRMemory,
    build_backbone,
    profile_flops,
    train_stream,
)

# ---------------------------------------------------------------------------
#  Directories – updated to iteration15 as per specification
# ---------------------------------------------------------------------------
RESULT_DIR = pathlib.Path(".research/iteration15")
FIG_DIR = pathlib.Path(".research/iteration15/images")

# ---------------------------------------------------------------------------
#  Environment sanity check – GPU required for the heavy models
# ---------------------------------------------------------------------------

def _gpu_guard():
    if not torch.cuda.is_available():
        sys.exit("[ERROR] GPU required – aborting as per experimental specification")


_gpu_guard()


# ---------------------------------------------------------------------------
#  Seeding helper (makes all libraries deterministic)
# ---------------------------------------------------------------------------

def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  Load configuration
# ---------------------------------------------------------------------------
CFG = yaml.safe_load(pathlib.Path("config/config.yaml").read_text())
SHARED = CFG["shared"]


# ---------------------------------------------------------------------------
#  Single experiment runner (one seed, one (method,budget) combo)
# ---------------------------------------------------------------------------

def _run_one(
    exp_cfg: Dict,
    seed: int,
):
    _seed_all(seed)
    device = "cuda"

    # -------------------------------------------------- dataset
    ds_name = exp_cfg["dataset"]
    if ds_name == "split_cifar100":
        train_tasks, test_set = build_split_cifar100(seed)
    elif ds_name == "cifar100_oneclass":
        train_tasks, test_set = build_cifar100_one_class(seed)
    else:
        raise ValueError(f"Unknown dataset '{ds_name}'")

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

    # -------------------------------------------------- loop over methods/budgets
    for method in exp_cfg.get("methods", [None]):
        if method is None:
            continue  # nothing to do
        for budget in exp_cfg.get("budgets_kb", [SHARED["budget_kb"]]):
            desc = (
                f"Dataset={ds_name}  Backbone={exp_cfg['backbone']}  "
                f"Method={method}  Budget={budget}kB  Seed={seed}"
            )
            print("\n" + "=" * len(desc) + f"\n{desc}\n" + "=" * len(desc))

            # ---------------- model construction --------------------------
            backbone = build_backbone(exp_cfg["backbone"]).to(device)
            classifier = ExpandingClassifier(256).to(device)
            params: List = list(backbone.parameters()) + list(classifier.parameters())

            losr = None
            buffer = None
            if method == "LOSR":
                losr = LOSRMemory(256, budget_kb=budget)
                losr.synth.to(device)
                params += list(losr.synth.parameters())
            elif method == "ER":
                buffer = ERBuffer(budget * 1024)
            # (Other baselines can be added analogously if needed)

            optimiser = torch.optim.AdamW(params, lr=SHARED["lr_grid"][0], weight_decay=1e-4)
            scaler = torch.cuda.amp.GradScaler()

            # ---------------- FLOPs (once) ---------------------------------
            flops_g = profile_flops(backbone)

            # ---------------- continual training --------------------------
            for subset in train_tasks:
                loader = DataLoader(
                    subset,
                    batch_size=SHARED["batch_size"],
                    shuffle=True,
                    num_workers=4,
                )
                for _ in range(SHARED["passes_per_task"]):
                    train_stream(
                        backbone,
                        classifier,
                        optimiser,
                        loader,
                        method,
                        losr,
                        buffer,
                        device,
                        scaler,
                    )

            # ---------------- evaluation ----------------------------------
            acc = evaluate(backbone, classifier, test_loader, device)
            if losr:
                mem_kb = losr.bytes() / 1024
            elif buffer:
                mem_kb = buffer.bytes() / 1024
            else:
                mem_kb = 0

            result = {
                "dataset": ds_name,
                "backbone": exp_cfg["backbone"],
                "method": method,
                "budget_kB": budget,
                "seed": seed,
                "accuracy": acc,
                "memory_kB": mem_kb,
                "train_FLOPs_G": flops_g,
            }

            out_path = RESULT_DIR / f"{method}_{budget}kB_seed{seed}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(result, out_path.open("w"), indent=2)
            print(json.dumps(result, indent=2))

            # ---------------- tiny diagnostic figure ----------------------
            plot_line(
                xs=[0, 1],
                ys=[0, acc * 100],
                xlab="",
                ylab="Accuracy %",
                title=f"Acc – {method}",
                fname=FIG_DIR / f"accuracy_{method}_{budget}.pdf",
            )


# ---------------------------------------------------------------------------
#  top-level entry point
# ---------------------------------------------------------------------------

def main():  # noqa: D401 – one word description unnecessary
    for exp in CFG["experiments"]:
        if exp["name"].startswith("EXP1"):
            continue  # EXP-1 is handled by unit-tests (not part of this refactor)
        for s in SHARED["seeds"]:
            _run_one(exp, s)


if __name__ == "__main__":
    main()
