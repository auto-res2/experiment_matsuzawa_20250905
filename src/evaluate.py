from __future__ import annotations
"""src/evaluate.py ––– metric computation & plotting utilities"""
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC / CI environments
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

sns.set(style="whitegrid", font_scale=1.2)

__all__ = [
    "compute_metrics",
    "generate_all_figures",
]

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, data):
    """Return standard train / val / test accuracy dictionary."""
    pred = logits.argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = int((pred[mask] == data.y[mask].to(pred.device)).sum())
        accs.append(correct / int(mask.sum()))
    train_acc, val_acc, test_acc = accs
    return {"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc}


def _annotate_bars(ax):
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=9)

def generate_all_figures(exp_name: str,
                         results: Dict[str, Any],
                         images_dir: Path) -> None:
    """Create a simple bar-plot per dataset showing test accuracy of all models."""
    images_dir.mkdir(parents=True, exist_ok=True)

    for dname, d_res in results.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        models = list(d_res.keys())
        accs = [d_res[m]["test_acc"] * 100 for m in models]
        sns.barplot(x=models, y=accs, ax=ax, palette="deep")
        _annotate_bars(ax)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_xlabel("Model")
        ax.set_title(f"{dname} – Test Accuracy")
        fname = images_dir / f"accuracy_{exp_name}_{dname}.pdf"
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)
        print(f"[FIGURE SAVED] {fname.relative_to(Path.cwd())}")
