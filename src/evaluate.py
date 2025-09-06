"""src/evaluate.py
-------------------------------------------------------------------------------
Model evaluation, statistics and plotting utilities.
Only lightweight seaborn / matplotlib functions live here so that they can be
mocked in unit tests when a display is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib

# Use a non-interactive backend for headless servers
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader

# ----------------------------- evaluation -------------------------------------

@torch.no_grad()
def accuracy(backbone: torch.nn.Module, clf: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    backbone.eval()
    clf.eval()
    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        feats = backbone(x)
        logits = clf(feats)
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(y.tolist())
    return accuracy_score(gts, preds)

# ----------------------------- persistence & plotting -------------------------

def save_json(result: dict, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as fp:
        json.dump(result, fp, indent=2)
    # echo for quick check in CI
    print(json.dumps(result, indent=2))


def barplot_accuracy(dataset: str, backbone: str, method: str, acc: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    sns.barplot(x=[method], y=[acc * 100])
    plt.ylabel("Final Avg. Accuracy [%]")
    plt.title(f"{dataset} – {backbone}")
    plt.text(0, acc * 100 + 0.5, f"{acc * 100:.1f}", ha="center")
    fname = out_dir / f"accuracy_{dataset}_{backbone}_{method}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Figure saved:", fname)
