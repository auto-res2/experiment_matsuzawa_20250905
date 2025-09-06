import json
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# ---------------------------------------------------------------------------
#  Evaluation metrics & visualisation helpers
# ---------------------------------------------------------------------------

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    if pred.ndim == 2:
        pred = pred.argmax(dim=-1)
    return float((pred == y).sum().item() / y.numel())


def row_diff(x: torch.Tensor) -> float:
    """Mean pair-wise Euclidean distance.  Sub-samples to keep memory bounded."""
    x = x.detach().cpu().float()
    if x.size(0) > 2048:
        idx = torch.randperm(x.size(0))[:2048]
        x = x[idx]
    dist = torch.pdist(x).mean().item()
    return dist


def group_distance_ratio(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().float()
    y = y.cpu().numpy()
    num_cls = int(y.max()) + 1
    intra, inter, cnt_intra = 0.0, 0.0, 0
    for c in range(num_cls):
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        intra += torch.pdist(x[idx]).mean().item()
        cnt_intra += 1
    if cnt_intra:
        intra /= cnt_intra
    # inter class – sample 256 nodes per class to bound cost
    for c1 in range(num_cls):
        for c2 in range(c1 + 1, num_cls):
            i1 = np.where(y == c1)[0][:256]
            i2 = np.where(y == c2)[0][:256]
            if len(i1) == 0 or len(i2) == 0:
                continue
            inter += ((x[i1][:, None, :] - x[i2][None, :, :]) ** 2).sum(-1).sqrt().mean().item()
    pairs = num_cls * (num_cls - 1) / 2
    inter /= max(pairs, 1)
    return inter / max(intra, 1e-9)


# ---------------------------------------------------------------------------
#  Simple line-plot writer (PDF)
# ---------------------------------------------------------------------------

def save_line(xs: List[int], ys: List[float], xlabel: str, ylabel: str, title: str, fname: str):
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    sns.lineplot(x=xs, y=ys, marker="o")
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
