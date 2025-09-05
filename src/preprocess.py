from __future__ import annotations

"""preprocess.py
===============================================================================
Data-handling utilities shared across the project (downloading, seeding, …).
Only a *very* small subset that is necessary for the CI smoke-test is
implemented. Anything more elaborate should be added as needed.
"""

from pathlib import Path
from typing import Any, Tuple
import random

import numpy as np
import torch
from wilds import get_dataset  # type: ignore

# --------------------------------------------------------------------------- #
# === constants ============================================================== #
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
# conform with the job description: store figures under .research/iteration6/images
FIG_DIR = ROOT / ".research" / "iteration6" / "images"

for _d in (DATA_DIR, OUTPUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# device / dtype helpers used throughout the codebase
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32  # keep things simple – no mixed-precision for tests

# --------------------------------------------------------------------------- #
# === seeding helper ========================================================= #
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]


def set_seed(seed: int = 0) -> None:  # noqa: D401 – imperative helper
    """Seed *all* randomness sources that could influence PyTorch experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour for cuDNN
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# === dataset loader ========================================================= #
# --------------------------------------------------------------------------- #

def load_dataset(exp_cfg: dict[str, Any]) -> Tuple[Any, Any]:  # noqa: D401 – simple helper
    """Load the train/validation split as utilised by WILDS datasets.

    For the purpose of the automated tests we only fetch the *metadata* split
    (this avoids downloading the full raw images which would blow up disk
    usage). A real experiment would, of course, load *all* data.
    """

    wilds_key = exp_cfg["dataset"].get("wilds_key")
    if wilds_key is None:
        raise ValueError("Configuration must specify 'wilds_key' inside 'dataset'.")

    dataset = get_dataset(dataset=wilds_key, download=True, root_dir=str(DATA_DIR))
    # Use WILDS standard splits – 0: train, 1: val, 2: test
    train_data = dataset.get_subset("train", transform=None)
    val_data = dataset.get_subset("val", transform=None)
    return train_data, val_data
