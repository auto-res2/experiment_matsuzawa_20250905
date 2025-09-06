"""
evaluate.py – metrics, statistics, plotting utilities
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import torch

# ----------------------------------------------------------------------------
#  Basic Metrics
# ----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute accuracy for classification logits."""
    if logits.ndim == 2:
        logits = logits.argmax(dim=-1)
    return float((logits == y).sum().item() / y.numel())


def row_diff(x: torch.Tensor, sample: int = 2048) -> float:
    x = x.detach().cpu()
    if x.size(0) > sample:
        idx = torch.randperm(x.size(0))[:sample]
        x = x[idx]
    return torch.pdist(x.float()).mean().item()


def group_distance_ratio(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().float()
    y = y.cpu()
    classes = torch.unique(y)
    intra, cnt = 0.0, 0
    for c in classes:
        idx = (y == c).nonzero(as_tuple=False).squeeze()
        if idx.numel() < 2:
            continue
        intra += torch.pdist(x[idx]).mean().item()
        cnt += 1
    intra = intra / max(cnt, 1)

    inter_vals: List[float] = []
    for c1, c2 in itertools.combinations(classes.tolist(), 2):
        i1 = (y == c1).nonzero(as_tuple=False).squeeze()[:256]
        i2 = (y == c2).nonzero(as_tuple=False).squeeze()[:256]
        if i1.numel() == 0 or i2.numel() == 0:
            continue
        inter_vals.append(((x[i1][:, None, :] - x[i2][None, :, :]) ** 2).sum(-1).sqrt().mean().item())
    inter = sum(inter_vals) / max(len(inter_vals), 1)
    return inter / max(intra, 1.0e-6)

# ----------------------------------------------------------------------------
#  Plot helpers (PDF only as per submission rules)
# ----------------------------------------------------------------------------

def lineplot(xs: List, ys: List, xlabel: str, ylabel: str, title: str, fname):
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    sns.lineplot(x=xs, y=ys, marker="o")
    for x_, y_ in zip(xs, ys):
        plt.text(x_, y_, f"{y_:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()
