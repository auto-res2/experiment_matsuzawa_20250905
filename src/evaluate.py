"""src.evaluate – evaluation utilities, metric computation, plotting and
miscellaneous helper functions that are shared across experiments."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # head-less backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Reproducibility --------------------------------------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Light-weight GPU timer -------------------------------------------------------
# -----------------------------------------------------------------------------

class CudaTimer:
    """Context-manager measuring (GPU) wall-clock time in milliseconds."""

    def __enter__(self):
        if torch.cuda.is_available():
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.cpu_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start.elapsed_time(self.end)
        else:
            self.elapsed_ms = (time.perf_counter() - self.cpu_start) * 1_000


# -----------------------------------------------------------------------------
# I/O helpers ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# -----------------------------------------------------------------------------
# Plotting ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _annotate(ax: plt.Axes):
    for line in ax.get_lines():
        x_data, y_data = line.get_xdata(), line.get_ydata()
        for x, y in zip(x_data, y_data):
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=6,
            )


def plot_line(
    x: List[float],
    ys: Dict[str, List[float]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> Path:
    plt.figure(figsize=(6, 4))
    for label, y in ys.items():
        plt.plot(x, y, marker="o", label=label)
    ax = plt.gca()
    _annotate(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    out_dir = Path(".research") / "iteration1" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / filename
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close()
    return outfile

# -----------------------------------------------------------------------------
# Core evaluation --------------------------------------------------------------
# -----------------------------------------------------------------------------

def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Top-1 accuracy computation."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, _ = model(x)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


__all__ = [
    "set_seed",
    "CudaTimer",
    "save_json",
    "plot_line",
    "evaluate",
]
