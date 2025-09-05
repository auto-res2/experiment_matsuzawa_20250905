"""src/evaluate.py
Evaluation helpers – plotting utilities, JSON serialisation, and higher-level
convenience wrappers that are used by *src.main*.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

LOGGER = logging.getLogger("pcd.evaluate")

ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / ".research" / "iteration1"
IMG_DIR = RESEARCH_DIR / "images"
RES_DIR = RESEARCH_DIR / "results"
for p in [IMG_DIR, RES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# JSON RESULTS -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def save_results_json(res: Dict, filename: str | Path):
    outfile = Path(filename)
    outfile.write_text(json.dumps(res, indent=2))
    LOGGER.info("Saved results ➜ %s", outfile.relative_to(ROOT))


# -----------------------------------------------------------------------------
# PLOTTING ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

def plot_learning_curves(train_loss: List[float], title: str, filename: str | Path):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 3))
    plt.plot(train_loss, label="Train Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    filepath = Path(filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved figure ➜ %s", filepath.relative_to(ROOT))


__all__ = ["save_results_json", "plot_learning_curves", "IMG_DIR", "RES_DIR"]