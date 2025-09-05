"""src/preprocess.py
Data downloading, synthetic dataset generation and DataLoader helpers.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

# -----------------------------------------------------------------------------
# Directory handling – `data` directory is created one level above src.
# -----------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

__all__ = [
    "DATA_DIR",
    "make_loader",
]

# -----------------------------------------------------------------------------
# 1.  Utility
# -----------------------------------------------------------------------------

def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------------
# 2.  Real datasets (WILDS, CelebA, CIFAR-10)
# -----------------------------------------------------------------------------

def get_waterbirds(split: str):
    try:
        from wilds import get_dataset  # type: ignore
    except ImportError:
        _die("Package `wilds` missing – install with `pip install wilds>=2.0`. ")

    dataset = get_dataset("waterbirds", root_dir=str(DATA_DIR), download=True)
    subset = dataset.get_subset(split)
    tfm = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.ConvertImageDtype(torch.float16)]
    )
    subset.transform = tfm
    return subset


def get_celeba(split: str):
    split_map = {"train": "train", "val": "valid", "test": "test"}
    if split not in split_map:
        _die(f"Bad split {split} for CelebA")

    tfm = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.ConvertImageDtype(torch.float16)]
    )

    ds = torchvision.datasets.CelebA(
        root=str(DATA_DIR / "celeba"),
        split=split_map[split],
        target_type=["attr"],
        download=True,
        transform=tfm,
    )

    class CelebHair(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
            self.attr_idx = 9  # Blond_Hair attribute

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, attrs = self.base[idx]
            label = int(attrs[self.attr_idx].item() == 1)
            return img, label

    return CelebHair(ds)


def get_cifar(split: str):
    tfm_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    tfm_test = T.ToTensor()

    if split == "train":
        return torchvision.datasets.CIFAR10(
            root=str(DATA_DIR / "cifar10"), train=True, download=True, transform=tfm_train
        )
    elif split in {"val", "test"}:
        full = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR / "cifar10"), train=(split == "val"), download=True, transform=tfm_test
        )
        if split == "val":
            train_len = int(0.9 * len(full))
            val_len = len(full) - train_len
            train_set, val_set = torch.utils.data.random_split(
                full, [train_len, val_len], generator=torch.Generator().manual_seed(42)
            )
            return val_set
        return full
    else:
        _die("CIFAR bad split")


# -----------------------------------------------------------------------------
# 3.  Synthetic DiffSpur dataset
# -----------------------------------------------------------------------------
import cv2

SHAPES = ["square", "triangle", "circle", "pentagon"]
COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


class DiffSpurDataset(torch.utils.data.Dataset):
    def __init__(self, *, n: int = 50_000, split: str = "train", seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)

        idx1, idx2 = int(0.8 * n), int(0.9 * n)
        self.all_data = [self._render_sample() for _ in range(n)]
        if split == "train":
            self.data = self.all_data[:idx1]
        elif split == "val":
            self.data = self.all_data[idx1:idx2]
        else:
            self.data = self.all_data[idx2:]

        self.tfm = T.ToTensor()

    def _render_sample(self):
        shape = random.choice(SHAPES)
        colour = random.choice(COLOURS)
        img = np.zeros((128, 128, 3), dtype=np.uint8) + np.array(colour, dtype=np.uint8)
        overlay = np.zeros_like(img)
        if shape == "square":
            cv2.rectangle(overlay, (32, 32), (96, 96), (255, 255, 255), -1)
        elif shape == "triangle":
            pts = np.array([[64, 32], [32, 96], [96, 96]], np.int32)
            cv2.fillPoly(overlay, [pts], (255, 255, 255))
        elif shape == "circle":
            cv2.circle(overlay, (64, 64), 32, (255, 255, 255), -1)
        else:  # pentagon
            pts = np.array([[64, 24], [32, 56], [40, 96], [88, 96], [96, 56]], np.int32)
            cv2.fillPoly(overlay, [pts], (255, 255, 255))
        img = cv2.addWeighted(overlay, 1, img, 0.7, 0)
        return img[..., ::-1], SHAPES.index(shape), COLOURS.index(colour)  # BGR -> RGB

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label_shape, colour_idx = self.data[idx]
        x = self.tfm(img)
        return x, label_shape, colour_idx


# -----------------------------------------------------------------------------
# 4.  DataLoader facade
# -----------------------------------------------------------------------------

def make_loader(dataset_name: str, split: str, batch_size: int, *, num_workers: int = 4):
    if dataset_name == "waterbirds":
        ds = get_waterbirds(split)
    elif dataset_name == "celeba":
        ds = get_celeba(split)
    elif dataset_name == "cifar10":
        ds = get_cifar(split)
    elif dataset_name == "diffspur":
        ds = DiffSpurDataset(split=split)
    else:
        _die(f"Dataset {dataset_name} not supported.")

    shuffle = split == "train"
    drop_last = shuffle
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
