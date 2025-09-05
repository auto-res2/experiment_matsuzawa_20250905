"""src/evaluate.py – evaluation helpers & visualisation utilities"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

# The directory in which we store publication-ready figures
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "evaluate",
    "plot_metric",
]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    task_id: int,
    device: torch.device,
) -> float:
    """Return classification accuracy on *loader* for *task_id*."""

    model.eval()
    # We always keep 100 classes for the metric so that different task splits
    # are comparable.  The unused classes simply never appear.
    acc_metric = MulticlassAccuracy(num_classes=100).to(device)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images, task_id)
        acc_metric.update(logits, labels)

    return acc_metric.compute().item()


def plot_metric(
    x: List[int],
    y: List[float],
    ylabel: str,
    title: str,
    fname: str,
) -> None:
    """Save a PDF line plot *fname* into FIG_DIR."""

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", label=ylabel)
    for xi, yi in zip(x, y):
        plt.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center")

    plt.xlabel("Task ID")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, ls=":", alpha=0.5)

    out_path = FIG_DIR / fname
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"[Figure saved] {out_path.name}")
    plt.close()