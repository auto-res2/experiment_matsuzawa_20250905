"""src/evaluate.py
Evaluation utilities: clean accuracy, PGD robustness and line-plot saving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------------------------
#                  CLEAN  VALIDATION / TEST  METRICS
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader
) -> Tuple[float, float, float, np.ndarray]:
    """Return (AvgAcc, Worst-Group-Acc, CorrGap, 2×2 confusion-matrix)."""
    model.eval()
    preds, labels, groups = [], [], []
    for imgs, y, g in loader:
        imgs = imgs.cuda(non_blocking=True)
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
        groups.append(g)

    y = torch.cat(labels)
    p = torch.cat(preds)
    g = torch.cat(groups)
    acc = (p == y).float()

    group_acc = [(acc[g == gi].mean().item()) for gi in range(4)]
    wg_acc = min(group_acc)

    land_major, land_minor = group_acc[0], group_acc[1]
    water_major, water_minor = group_acc[3], group_acc[2]
    corr_gap = 0.5 * (abs(land_major - land_minor) + abs(water_major - water_minor))

    cm = confusion_matrix(y, p, labels=[0, 1])
    return acc.mean().item(), wg_acc, corr_gap, cm

# -----------------------------------------------------------------------------
#                    VERY  LIGHTWEIGHT  PGD-L∞  ATTACK
# -----------------------------------------------------------------------------

def _pgd_attack(
    model: torch.nn.Module, imgs: torch.Tensor, labels: torch.Tensor, eps: float, steps: int
) -> torch.Tensor:
    delta = torch.zeros_like(imgs, device=imgs.device, requires_grad=True)
    for _ in range(steps):
        outputs = model(imgs + delta)
        F.cross_entropy(outputs, labels).backward()
        grad = delta.grad.detach()
        delta.data = (delta + 0.01 * torch.sign(grad)).clamp(-eps, eps)
        delta.grad.zero_()
    return delta.detach()

@torch.no_grad()
def evaluate_pgd(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, eps: float, steps: int
) -> float:
    model.eval()
    total, correct = 0, 0
    for imgs, labels, _ in loader:
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        delta = _pgd_attack(model, imgs, labels, eps, steps)
        preds = model(imgs + delta).argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / total

# -----------------------------------------------------------------------------
#                                PLOTTING
# -----------------------------------------------------------------------------

def save_lineplot(
    x: List[int],
    ys: Dict[str, List[float]],
    ylabel: str,
    name: str,
    fig_dir: Path,
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    for label, y in ys.items():
        plt.plot(x, y, marker="o", label=label)
        for xv, yv in zip(x, y):
            plt.annotate(f"{yv:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    fn = fig_dir / f"{name}.pdf"
    plt.savefig(fn, bbox_inches="tight")
    print(f"[Fig ] saved {fn.relative_to(Path('.'))}")
    plt.close()
