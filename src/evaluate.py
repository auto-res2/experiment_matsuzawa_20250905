from __future__ import annotations

"""evaluate.py
================
Light-weight utilities for evaluation, statistical analysis & plotting.
Currently only a pretty line-plot function is required.
"""

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-order
import seaborn as sns  # noqa: E402  pylint: disable=wrong-import-order

from .preprocess import FIG_DIR


def save_lineplot(values: List[float], title: str, ylabel: str, filename: str) -> str:
    """Save a *.pdf line-plot to the shared figures directory and return the file-name."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(values) + 1), values, marker="o", label=title)
    for i, v in enumerate(values, start=1):
        plt.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(-5, 5))
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"{filename}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path.name