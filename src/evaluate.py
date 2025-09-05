from pathlib import Path
from typing import List

import matplotlib

# Use a non-interactive backend suitable for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 – backend must be set before import

# ----------------------------------------------------------------------------
# Image output directory (specification: `.research/iteration3/images`)
# ----------------------------------------------------------------------------
IMG_DIR = Path(".research/iteration3/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)


def save_line_plot(xs: List[int], ys: List[float], title: str, ylabel: str, fname: str):
    """Save a simple line plot under the research images directory."""
    fpath = IMG_DIR / fname
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label=title)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center")
    ax.set_xlabel("Depth (L)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.savefig(fpath, bbox_inches="tight", format="pdf")
    plt.close(fig)
