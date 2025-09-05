from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless backend for server/CI environments
import matplotlib.pyplot as plt
import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Paths (import lazily to avoid circular dependency)
# ---------------------------------------------------------------------------
from .preprocess import FIG_DIR

__all__ = [
    "average_pairwise_distance",
    "effective_rank",
    "save_line_plot",
]

# ---------------------------------------------------------------------------
# Metric utilities (GPU-friendly)
# ---------------------------------------------------------------------------

def average_pairwise_distance(h: Tensor) -> float:
    """Average pairwise Euclidean distance (APD)."""
    with torch.no_grad():
        h = h.float()
        norm = (h * h).sum(-1, keepdim=True)
        dist2 = (norm + norm.t() - 2.0 * h @ h.t()).clamp_min_(0.0)
        idx = torch.triu_indices(h.size(0), h.size(0), offset=1, device=h.device)
        return dist2[idx[0], idx[1]].sqrt().mean().item()


def effective_rank(h: Tensor, k: int = 128) -> float:
    """Shannon-entropy based effective rank (truncated at k singular values)."""
    with torch.no_grad():
        _, s, _ = torch.linalg.svd(h.float(), full_matrices=False)
        s = s[:k]
        p = s / s.sum()
        return torch.exp(-(p * p.log()).sum()).item()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_line_plot(xs: List[int], ys: List[float], title: str, ylab: str, fname: str) -> None:
    """Save a simple line plot as a PDF in FIG_DIR."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
    ax.set_xlabel("Depth (L)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    (FIG_DIR / fname).with_suffix(".pdf")
    plt.savefig(FIG_DIR / fname, bbox_inches="tight", format="pdf")
    plt.close(fig)
