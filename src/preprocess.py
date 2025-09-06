"""src/preprocess.py – data download, checksum and DataLoader helpers"""
from __future__ import annotations

# std -----------------------------------------------------------------------
import hashlib, os, random, tarfile, pathlib
from typing import Tuple, List

# third-party ---------------------------------------------------------------
import requests
import torchvision
import torch
from torch.utils.data import Subset

# ============================================================================
# Constants ------------------------------------------------------------------

DATA_ROOT = "data"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"

Path = pathlib.Path

# ============================================================================
# Utility helpers ------------------------------------------------------------

def _md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, target: Path, md5: str) -> None:
    if target.exists() and _md5sum(target) == md5:
        return
    print(f"[Download] {url} -> {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with target.open("wb") as fp:
            for chunk in r.iter_content(chunk_size=8192):
                fp.write(chunk)
    if _md5sum(target) != md5:
        raise RuntimeError("Checksum mismatch for " + str(target))

# ============================================================================
# Dataset builders -----------------------------------------------------------

def build_split_cifar100(seed: int) -> Tuple[List[Subset], torch.utils.data.Dataset]:
    """Return a list of 20 task-datasets (5 classes each) and the full test set."""
    data_root = Path(DATA_ROOT)
    data_root.mkdir(parents=True, exist_ok=True)
    arc = data_root / "cifar-100-python.tar.gz"
    _download(CIFAR100_URL, arc, CIFAR100_MD5)

    if not (data_root / "cifar-100-python").exists():
        with tarfile.open(arc) as tf:
            tf.extractall(data_root)

    tf_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )
    tf_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=False, transform=tf_train
    )
    test = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=False, transform=tf_test
    )

    cls_order = list(range(100))
    random.Random(seed).shuffle(cls_order)
    tasks = [cls_order[i * 5 : (i + 1) * 5] for i in range(20)]

    train_stream = []
    for cls in tasks:
        idx = [i for i, y in enumerate(train.targets) if y in cls]
        train_stream.append(Subset(train, idx))

    return train_stream, test
