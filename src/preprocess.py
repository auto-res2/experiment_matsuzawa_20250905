from __future__ import annotations

"""src/preprocess.py
-------------------------------------------------------------------------------
Data-handling, downloads and reproducibility utilities.
"""

import hashlib
import random
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List
from types import SimpleNamespace

import numpy as np
import requests
import torch
import torchvision
from torch.utils.data import DataLoader, Subset

# -----------------------------------------------------------------------------
# 1.  GLOBAL PATHS
# -----------------------------------------------------------------------------

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Updated research folder structure to comply with the prompt requirements
RESEARCH_ROOT = Path(".research") / "iteration3"
IMAGES_DIR = RESEARCH_ROOT / "images"
RESULTS_DIR = RESEARCH_ROOT
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2.  DOWNLOAD HELPERS
# -----------------------------------------------------------------------------

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"

MINI_IMAGENET_URL = "https://figshare.com/ndownloader/files/40552732"
MINI_IMAGENET_ZIP = "mini-imagenet.zip"
MINI_IMAGENET_SHA1 = "d85c8b055ff3e4b34c1a0533e87e817939e6f145"


def _hash_file(path: Path, algorithm: str = "md5", chunk_sz: int = 8192) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_sz)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, expected_hash: str | None = None, algo: str = "md5") -> None:
    """Download *url* to *dest* if missing and verify checksum."""
    if dest.exists() and (expected_hash is None or _hash_file(dest, algo) == expected_hash):
        return  # already downloaded & healthy

    print(f"[Download] Fetching {url} → {dest}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with dest.open("wb") as fp:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

    if expected_hash and _hash_file(dest, algo) != expected_hash:
        raise RuntimeError(f"Checksum mismatch for {dest}")


# -----------------------------------------------------------------------------
# 3.  DATASET BUILDERS
# -----------------------------------------------------------------------------


def _download_cifar100() -> Path:
    """Download and extract the original CIFAR-100 python archive."""
    arc = DATA_ROOT / "cifar-100-python.tar.gz"
    download(CIFAR100_URL, arc, CIFAR100_MD5, "md5")
    extract_dir = DATA_ROOT / "cifar-100-python"
    if not extract_dir.exists():
        with tarfile.open(arc) as tf:
            tf.extractall(DATA_ROOT)
    return extract_dir


def _download_mini_imgnet() -> Path:
    arc = DATA_ROOT / MINI_IMAGENET_ZIP
    download(MINI_IMAGENET_URL, arc, MINI_IMAGENET_SHA1, "sha1")
    extract_dir = DATA_ROOT / "mini-imagenet"
    if not extract_dir.exists():
        with zipfile.ZipFile(arc) as zf:
            zf.extractall(DATA_ROOT)
    return extract_dir


# ------------------------- CIFAR-100 benchmark --------------------------------

def build_cifar100_benchmark(val_ratio: float = 0.05, seed: int = 0):
    """Mimic Avalanche's SplitCIFAR100 (20 × 5-class experiences).

    Only the parts that are actually used by *src.main* are implemented: a
    20-element *train_stream* (list of Subset objects) and a *test_stream*
    containing a single element whose ``dataset`` attribute points to the full
    CIFAR-100 test set.  This avoids the heavyweight ``avalanche-lib`` package
    which currently pulls in the incompatible *proxsuite* dependency.
    """
    _download_cifar100()

    tf_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    tf_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    train_set = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=tf_train)
    test_set = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=tf_test)

    # ----- class order & per-experience split --------------------------------
    class_order = list(range(100))
    random.Random(seed).shuffle(class_order)
    exp_classes = [class_order[i * 5: (i + 1) * 5] for i in range(20)]

    train_stream: List[Subset] = []
    for cls in exp_classes:
        idx = [i for i, lbl in enumerate(train_set.targets) if lbl in cls]
        train_stream.append(Subset(train_set, idx))

    # A tiny shim so that *main.py* can access ``bench.test_stream[0].dataset``
    test_stream = [SimpleNamespace(dataset=test_set)]

    return SimpleNamespace(train_stream=train_stream, test_stream=test_stream)


# ------------------------- Mini-ImageNet benchmark ----------------------------

class _MiniImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root: Path, split: str, transform):
        super().__init__(root / split, transform=transform)


def build_mini_imgnet_benchmark(val_ratio: float = 0.05, seed: int = 0):
    root = _download_mini_imgnet()
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(92),
        torchvision.transforms.CenterCrop(84),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_set = _MiniImageNet(root, "train", tf)
    test_set = _MiniImageNet(root, "test", tf)

    cls_order = list(range(100))
    random.Random(seed).shuffle(cls_order)
    exp_classes = [cls_order[i * 5 : (i + 1) * 5] for i in range(20)]

    experiences = []
    for cls in exp_classes:
        idx = [i for i, (_, y) in enumerate(train_set.samples) if y in cls]
        experiences.append(Subset(train_set, idx))

    return {"train_stream": experiences, "test_set": test_set}

# -----------------------------------------------------------------------------
# 4.  REPRODUCIBILITY
# -----------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
