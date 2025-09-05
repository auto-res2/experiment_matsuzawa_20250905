"""src/evaluate.py – evaluation helpers & visualisation utilities"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
#  Save all figures inside the directory mandated by the instructions
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / ".research" / "iteration2" / "images"
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
    """Return classification accuracy on *loader* for *task_id*.

    A hand-rolled metric is used instead of ``torchmetrics`` so that we do not
    have to know ``num_classes`` beforehand (it differs across tasks).  This
    keeps the evaluation routine generic and avoids shape-mismatch errors.
    """

    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images, task_id)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total > 0 else 0.0


def plot_metric(
    x: List[int],
    y: List[float],
    ylabel: str,
    title: str,
    fname: str,
) -> None:
    """Save a PDF line plot *fname* into ``FIG_DIR``."""

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
    print(f"[Figure saved] {out_path.relative_to(ROOT)}")
    plt.close()