"""src/evaluate.py
Evaluation helpers (plots, metrics) used by experiments.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import pathlib
from typing import Dict


def plot_bar(values: Dict[str, float], title: str, ylabel: str, filename: str):
    names = list(values.keys())
    vals = list(values.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color="tab:blue")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
