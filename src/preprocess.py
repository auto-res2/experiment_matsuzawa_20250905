from __future__ import annotations

"""preprocess.py
================
Data-loading helpers, reproducibility utilities & global constants shared
between the other modules.
"""

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torchvision import transforms as T
from wilds import get_dataset

# --------------------------------------------------------------------------- #
# === global constants / directories ======================================= #
# --------------------------------------------------------------------------- #
SEEDS = [0, 1, 2, 3, 4]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / ".research" / "iteration1" / "images"
for d in (DATA_DIR, OUTPUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# === reproducibility helper =============================================== #
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# === torchvision transforms =============================================== #
# --------------------------------------------------------------------------- #
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --------------------------------------------------------------------------- #
# === dataset loaders using WILDS ========================================== #
# --------------------------------------------------------------------------- #

def _load_wilds_subset(name: str, split: str):
    ds = get_dataset(name, root_dir=DATA_DIR, download=True)
    return ds.get_subset(split, transform=TRANSFORM)


def load_waterbirds(split: str):
    return _load_wilds_subset("waterbirds", split)


def load_celeba(split: str):  # hair-colour as spurious attribute
    return _load_wilds_subset("celebA", split)


def load_dataset(cfg: Dict[str, Any]):
    name: str = cfg["dataset"]["name"]
    if name == "waterbirds":
        return load_waterbirds("train"), load_waterbirds("val")
    if name == "celeba_hair":
        return load_celeba("train"), load_celeba("val")
    raise ValueError(f"Unsupported dataset {name}")