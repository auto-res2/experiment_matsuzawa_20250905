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
from torchvision import transforms  # new import

# --------------------------------------------------------------------------- #
# === constants ============================================================== #
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
# Updated per instructions: store figures under .research/iteration10/images
FIG_DIR = ROOT / ".research" / "iteration10" / "images"

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

# A very lightweight transform pipeline shared across all datasets
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Mapping to correct WILDS keys when spelling/casing differs in config
_WILDS_ALIAS = {
    "celeba": "celebA",
}


def _canonical_wilds_key(key: str) -> str:
    """Return a WILDS dataset key that `wilds.get_dataset` recognises.

    The function performs a case-insensitive lookup and falls back to a fixed
    alias-table for special cases like `celebA` whose mixed-case spelling is
    easy to get wrong in configuration files.
    """
    key_lower = key.lower()
    if key_lower in _WILDS_ALIAS:
        return _WILDS_ALIAS[key_lower]

    # Attempt case-insensitive match against the official list to avoid future
    # alias maintenance.
    from wilds import supported_datasets  # imported lazily to keep import cost low

    for official in supported_datasets:
        if official.lower() == key_lower:
            return official

    # If we reach this point the key is unknown – propagate the original value
    return key


def load_dataset(exp_cfg: dict[str, Any]) -> Tuple[Any, Any]:  # noqa: D401 – simple helper
    """Load the train/validation split as utilised by WILDS datasets.

    For the purpose of the automated tests we only fetch the *metadata* split
    (this avoids downloading the full raw images which would blow up disk
    usage). A real experiment would, of course, load *all* data.
    """

    wilds_key_cfg = exp_cfg["dataset"].get("wilds_key")
    if wilds_key_cfg is None:
        raise ValueError("Configuration must specify 'wilds_key' inside 'dataset'.")

    wilds_key = _canonical_wilds_key(wilds_key_cfg)

    dataset = get_dataset(dataset=wilds_key, download=True, root_dir=str(DATA_DIR))
    # Use WILDS standard splits – 0: train, 1: val, 2: test
    train_data = dataset.get_subset("train", transform=DEFAULT_TRANSFORM)
    val_data = dataset.get_subset("val", transform=DEFAULT_TRANSFORM)
    return train_data, val_data
