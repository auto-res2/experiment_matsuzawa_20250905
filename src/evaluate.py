"""src/evaluate.py
Evaluation helpers (calibration error and simple plotting).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchmetrics import CalibrationError

__all__ = [
    "expected_calibration_error",
    "save_lineplot",
]

# -----------------------------------------------------------------------------
# 1.  Calibration error
# -----------------------------------------------------------------------------

def expected_calibration_error(model: torch.nn.Module, loader: torch.utils.data.DataLoader, *, device: str = "cuda") -> float:
    """Compute Expected Calibration Error (ECE) over the loader."""
    metric = CalibrationError(n_bins=15, norm="l1").to(device)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            metric.update(logits.softmax(-1), y)
    return float(metric.compute())


# -----------------------------------------------------------------------------
# 2.  Simple line plot for learning curves
# -----------------------------------------------------------------------------

def save_lineplot(xs: List[int], ys: List[float], title: str, ylabel: str, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
