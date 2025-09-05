"""
evaluate.py – evaluation metrics and plotting helpers
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mutual_info_score

matplotlib.use("Agg")

_IMAGES_DIR = Path(".research/iteration1/images")
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────────────

def accuracy(logits, labels):
    return accuracy_score(labels.cpu(), logits.argmax(dim=1).cpu())


def group_distance_ratio(emb, labels):
    with torch.no_grad():
        labels_np = labels.cpu().numpy()
        emb_cpu = emb.cpu()
        intra = inter = cnt_intra = cnt_inter = 0.0
        for i in range(emb_cpu.size(0)):
            for j in range(i + 1, emb_cpu.size(0)):
                d = (emb_cpu[i] - emb_cpu[j]).pow(2).sum().sqrt().item()
                if labels_np[i] == labels_np[j]:
                    intra += d
                    cnt_intra += 1
                else:
                    inter += d
                    cnt_inter += 1
        if cnt_intra == 0 or cnt_inter == 0:
            return 0.0
        return (inter / cnt_inter) / (intra / cnt_intra + 1e-9)


def instance_information_gain(emb, labels, n_clusters: int):
    with torch.no_grad():
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(emb.cpu())
        mi = mutual_info_score(labels.cpu().numpy(), kmeans.labels_)
    return mi / np.log(n_clusters)


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_line(
    xs: List[float],
    ys_dict: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
):
    plt.figure(figsize=(6, 4))
    for label, ys in ys_dict.items():
        plt.plot(xs, ys, marker="o", label=label)
        for x, y in zip(xs, ys):
            plt.annotate(
                f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8
            )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    full_path = _IMAGES_DIR / filename
    plt.savefig(full_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"Figure saved: {full_path}")
