"""
evaluate.py – model evaluation & quick plotting utilities
"""
from __future__ import annotations

import json
import pathlib
from typing import Sequence

import matplotlib
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

# matplotlib in non-interactive backend to allow head-less execution
matplotlib.use("pdf")

Path = pathlib.Path

# ---------------------------------------------------------------------------
#  Accuracy evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(backbone, classifier, loader, device: str = "cuda") -> float:
    """Return accuracy on *loader* (0-1 range)"""

    backbone.eval()
    classifier.eval()

    preds: Sequence[int] = []
    gts: Sequence[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = classifier(backbone(x))
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(y.tolist())

    return accuracy_score(gts, preds)


# ---------------------------------------------------------------------------
#  Minimal line plot helper – stores PDF in research directory
# ---------------------------------------------------------------------------

def plot_line(xs, ys, xlab: str, ylab: str, title: str, fname: Path) -> None:
    plt.figure(figsize=(6, 3))
    sns.lineplot(x=xs, y=ys, marker="o")
    for x_val, y_val in zip(xs, ys):
        plt.text(x_val, y_val, f"{y_val:.1f}")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("[Figure]", fname)
