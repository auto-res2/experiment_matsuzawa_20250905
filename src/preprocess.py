"""src/preprocess.py
Datasets and data-loading utilities (TorchVision + Avalanche).
"""
from __future__ import annotations
from typing import List, Tuple
import pathlib, random

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Avalanche is heavier; we guard import to provide graceful error message
try:
    from avalanche.benchmarks.classic import SplitCIFAR100
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Avalanche is required (pip install avalanche-lib)") from exc

import torchvision.datasets as tvd

# -----------------------------------------------------------------------------
#                       Common transformations
# -----------------------------------------------------------------------------

_CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR_STD = (0.2675, 0.2565, 0.2761)


def _common_transforms(train: bool):
    aug: List[T.transforms] = []
    if train:
        aug += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    aug += [T.ToTensor(), T.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    return T.Compose(aug)

# -----------------------------------------------------------------------------
#                       Split-CIFAR-100 benchmark
# -----------------------------------------------------------------------------

def get_split_cifar100(batch_size: int = 128, *, seed: int = 0):
    train_tf = _common_transforms(True)
    test_tf = _common_transforms(False)
    benchmark = SplitCIFAR100(
        n_experiences=20,
        seed=seed,
        return_task_id=True,
        train_transform=train_tf,
        eval_transform=test_tf,
    )
    return benchmark

# -----------------------------------------------------------------------------
#                       Streaming CIFAR loader
# -----------------------------------------------------------------------------

def get_streaming_cifar(batch_size: int = 10):
    tf = _common_transforms(True)
    ds = tvd.CIFAR100(root="data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
