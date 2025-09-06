"""src/evaluate.py – evaluation utilities and visualisation"""
from __future__ import annotations

import json, pathlib
from typing import Sequence

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score

Path = pathlib.Path

# ============================================================================
# Evaluation -----------------------------------------------------------------

@torch.no_grad()
def evaluate(backbone, clf, loader, device: str = "cuda") -> float:
    backbone.eval()
    clf.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = clf(backbone(x))
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(y.tolist())
    return accuracy_score(gts, preds)


# ============================================================================
# Plotting -------------------------------------------------------------------

def plot_curves(
    xs: Sequence[float],
    ys: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    fname: Path,
) -> None:
    plt.figure(figsize=(5, 3))
    sns.lineplot(x=xs, y=ys, marker="o")
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.1f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Figure saved:", fname)
