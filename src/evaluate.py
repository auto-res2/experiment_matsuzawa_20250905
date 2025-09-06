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
    """A simple white-box PGD-L∞ attack.

    Notes
    -----
    • Implements the *untargeted* variant with fixed step-size 0.01.
    • Returned perturbation *delta* is **detached** so that the caller can
      safely use it in a no-grad context.
    """
    delta = torch.zeros_like(imgs, device=imgs.device, requires_grad=True)
    step_size = 0.01  # hard-coded as in the original released baseline

    for _ in range(steps):
        # forward & backward ---------------------------------------------------
        outputs = model(imgs + delta)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # gradient sign step ---------------------------------------------------
        grad = delta.grad.detach()
        delta.data = (delta + step_size * torch.sign(grad)).clamp(-eps, eps)

        # reset gradient for the next iteration -------------------------------
        delta.grad.zero_()

    return delta.detach()


def evaluate_pgd(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, eps: float, steps: int
) -> float:
    """Report classification accuracy under an untargeted PGD-L∞ attack."""

    # We need gradients for the adversarial search, but not for the forward
    # pass used to *measure* accuracy.  Hence, we selectively enable / disable
    # autograd instead of decorating the whole function with @torch.no_grad().

    model.eval()
    total, correct = 0, 0

    for imgs, labels, _ in loader:
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # 1) craft the adversarial perturbation (requires grad)
        with torch.enable_grad():
            delta = _pgd_attack(model, imgs, labels, eps, steps)

        # 2) evaluate robustness without tracking gradients -------------------
        with torch.no_grad():
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
