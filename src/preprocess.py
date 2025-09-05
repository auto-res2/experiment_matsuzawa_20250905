"""
src/preprocess.py – dataset download & class-incremental stream builder
Currently supports CIFAR-100 only (20×5 split by default).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torchvision
from torch.utils.data import Dataset, random_split
from torchvision import transforms as T
from torchvision.datasets import CIFAR100


class ToHalf(torchvision.transforms.Lambda):
    def __init__(self):
        super().__init__(lambda x: x.half())


def build_stream(cfg: dict) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    name = cfg["name"].lower()
    if name == "cifar100":
        return _cifar100_stream(cfg)
    raise RuntimeError(f"Dataset {name} not implemented.")


# -----------------------------------------------------------------------------
#  CIFAR-100   20 tasks × 5 classes (default)
# -----------------------------------------------------------------------------

def _cifar100_stream(cfg):
    root = Path("data") / "cifar100"
    train = CIFAR100(root=root, train=True, download=True)
    test = CIFAR100(root=root, train=False, download=True)

    train.transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, 4),
        T.ToTensor(),
        ToHalf(),
    ])
    test.transform = T.Compose([T.ToTensor(), ToHalf()])

    num_tasks = cfg["num_tasks"]
    cls_per_task = cfg["classes_per_task"]
    assert num_tasks * cls_per_task == 100

    tasks = [list(range(i * cls_per_task, (i + 1) * cls_per_task)) for i in range(num_tasks)]

    def _subset(dataset, cls):
        idx = [i for i, (_, y) in enumerate(dataset) if y in cls]
        return torch.utils.data.Subset(dataset, idx)

    train_stream, val_stream, test_stream = [], [], []
    for cls in tasks:
        full = _subset(train, cls)
        val_len = int(0.1 * len(full))
        train_len = len(full) - val_len
        tr_split, val_split = random_split(full, [train_len, val_len])
        train_stream.append(tr_split)
        val_stream.append(val_split)
        test_stream.append(_subset(test, cls))

    return train_stream, val_stream, test_stream
