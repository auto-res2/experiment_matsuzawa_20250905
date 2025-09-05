from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from wilds import get_dataset

__all__ = ["build_dataloader"]

_DATA_DIR = Path.cwd() / "data"
_DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
#  Natural image data sets (Waterbirds, CelebA-Hair, CIFAR-10)
# --------------------------------------------------------------------------


def _waterbirds(split: str) -> Dataset:
    ds = get_dataset("waterbirds", root_dir=str(_DATA_DIR), download=True)
    subset = ds.get_subset(split)

    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
    ])

    class _WB(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, y, _ = self.base[idx]
            return tfm(img), int(y)

    return _WB(subset)


def _celeba(split: str) -> Dataset:
    split_map = {"train": "train", "val": "valid", "test": "test"}
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
    ])
    ds = torchvision.datasets.CelebA(
        root=str(_DATA_DIR / "celeba"),
        split=split_map[split],
        target_type=["attr"],
        download=True,
        transform=tfm,
    )

    class _Hair(Dataset):
        def __init__(self, base):
            self.base = base
            self.attr_idx = 9  # *Blond Hair*

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, attrs = self.base[idx]
            y = int(attrs[self.attr_idx] == 1)
            return img, y

    return _Hair(ds)


def _cifar(split: str) -> Dataset:
    tfm_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    tfm_test = T.ToTensor()

    if split == "train":
        return torchvision.datasets.CIFAR10(
            root=str(_DATA_DIR / "cifar"), train=True, download=True, transform=tfm_train
        )
    return torchvision.datasets.CIFAR10(
        root=str(_DATA_DIR / "cifar"), train=False, download=True, transform=tfm_test
    )


# --------------------------------------------------------------------------
#  Synthetic *DiffSpur* data set used only for causal discovery
# --------------------------------------------------------------------------

_SHAPES = ["square", "triangle", "circle", "pentagon"]
_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


class _DiffSpur(Dataset):
    def __init__(self, n: int = 100_000, split: str = "train", img_size: int = 128, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)

        # deterministic split --------------------------------------------------
        idx1, idx2 = int(0.8 * n), int(0.9 * n)
        self.data = [self._render(img_size) for _ in range(n)]
        if split == "train":
            self.data = self.data[:idx1]
        elif split == "val":
            self.data = self.data[idx1:idx2]
        else:
            self.data = self.data[idx2:]
        self._to_tensor = T.ToTensor()

    # ---------------------------------------------------------------------
    def _render(self, s: int):
        shape = random.choice(_SHAPES)
        colour = random.choice(_COLOURS)
        img = np.zeros((s, s, 3), np.uint8) + np.array(colour, dtype=np.uint8)
        overlay = np.zeros_like(img)

        if shape == "square":
            cv2.rectangle(overlay, (s // 4, s // 4), (3 * s // 4, 3 * s // 4), (255, 255, 255), -1)
        elif shape == "triangle":
            pts = np.array([[s // 2, s // 4], [s // 4, 3 * s // 4], [3 * s // 4, 3 * s // 4]], np.int32)
            cv2.fillPoly(overlay, [pts], (255, 255, 255))
        elif shape == "circle":
            cv2.circle(overlay, (s // 2, s // 2), s // 4, (255, 255, 255), -1)
        else:  # pentagon
            pts = np.array(
                [
                    [s // 2, s // 5],
                    [s // 4, 7 * s // 15],
                    [5 * s // 16, 4 * s // 5],
                    [11 * s // 16, 4 * s // 5],
                    [3 * s // 4, 7 * s // 15],
                ],
                np.int32,
            )
            cv2.fillPoly(overlay, [pts], (255, 255, 255))

        img = cv2.addWeighted(overlay, 1, img, 0.7, 0)
        return img[..., ::-1], _SHAPES.index(shape), _COLOURS.index(colour)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, y, c = self.data[idx]
        return self._to_tensor(img), y, c


# --------------------------------------------------------------------------
#  Public interface
# --------------------------------------------------------------------------

def build_dataloader(
    cfg: "omegaconf.DictConfig",  # quoted import to avoid hard dependency here
    split: str,
    batch_size: int,
    workers: int = 4,
):
    name = cfg.dataset.name
    if name == "waterbirds":
        ds = _waterbirds(split)
    elif name == "celeba":
        ds = _celeba(split)
    elif name == "cifar10":
        ds = _cifar(split)
    elif name == "diffspur":
        ds = _DiffSpur(n=cfg.dataset.n_images, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
