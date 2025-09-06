"""
evaluate.py
===========
Evaluation utilities: metrics, plotting helpers, simple statistics.  Importable
from both `train.py` and `main.py`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score
import matplotlib as mpl

mpl.use("Agg")  # head-less backend (important for cluster / CI runs)
import matplotlib.pyplot as plt

__all__ = [
    "accuracy",
    "effective_rank",
    "group_distance_ratio",
    "line_plot",
]

# ---------------------------------------------------------------------------
# 1.  Metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).cpu().numpy()
    return float(accuracy_score(labels.cpu().numpy(), preds))


def effective_rank(emb: torch.Tensor) -> float:
    with torch.no_grad():
        u, s, _ = torch.linalg.svd(emb - emb.mean(0), full_matrices=False)
        p = s / s.sum()
        er = torch.exp(-(p * torch.log(p + 1e-12)).sum()) / len(s)
    return float(er.item())


def group_distance_ratio(emb: torch.Tensor, labels: torch.Tensor) -> float:
    emb_np = emb.cpu().numpy()
    y_np = labels.cpu().numpy()
    overall = np.linalg.norm(emb_np[:, None, :] - emb_np[None, :, :], axis=-1)
    within, across = [], []
    for i in range(len(y_np)):
        for j in range(len(y_np)):
            if i == j:
                continue
            (within if y_np[i] == y_np[j] else across).append(overall[i, j])
    return float(np.mean(across) / (np.mean(within) + 1e-9))

# ---------------------------------------------------------------------------
# 2.  Plotting
# ---------------------------------------------------------------------------

def line_plot(
    xs: List,
    ys_dict: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    pdf_path: Path | str,
):
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for label, ys in ys_dict.items():
        plt.plot(xs, ys, label=label, marker="o")
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(pdf_path, bbox_inches="tight")
    finally:
        plt.close()
