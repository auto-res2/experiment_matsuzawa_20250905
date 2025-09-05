"""src/preprocess.py
Data download, preprocessing and continual stream creation.
All routines are purely functional and *must* receive the config dict.
"""
from __future__ import annotations

import random
import tarfile
from pathlib import Path
from typing import List, Tuple

import requests
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split


# ----------------------------------------------------------------------
#  Helper – robust HTTP download with resume support
# ----------------------------------------------------------------------


def _download(url: str, dst: Path):
    if dst.exists():
        return
    print(f"[INFO] Downloading {url} …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=2 ** 20):
                if chunk:
                    f.write(chunk)
    print(f"[INFO] Saved to {dst}")


# ----------------------------------------------------------------------
#  CIFAR-100 specific preparation
# ----------------------------------------------------------------------


def _prepare_cifar(root: Path, cfg):  # noqa: ANN001
    tar_path = root / "cifar-100-python.tar.gz"
    _download(cfg["dataset"]["url"], tar_path)
    if not (root / "cifar-100-python").exists():
        print("[INFO] Extracting CIFAR-100 tarball …")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=root)


# ----------------------------------------------------------------------
#  Class-incremental split helpers
# ----------------------------------------------------------------------


def _split_by_class(ds, num_tasks: int, classes_per_task: int):  # noqa: ANN001
    idx_per_task: List[List[int]] = [[] for _ in range(num_tasks)]
    for i, (_, y) in enumerate(ds):
        idx_per_task[y // classes_per_task].append(i)
    return idx_per_task


# ----------------------------------------------------------------------
#  Data augmentation
# ----------------------------------------------------------------------


def _build_transforms(cfg, train: bool):  # noqa: ANN001
    size = cfg["dataset"]["img_size"]
    if train:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(size, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float16),
                torchvision.transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float16),
            torchvision.transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )


# ----------------------------------------------------------------------
#  Build the continual stream – returns train/val/test loader lists
# ----------------------------------------------------------------------


def build_stream(cfg) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:  # noqa: ANN001
    name = cfg["dataset"]["name"].lower()
    if name != "cifar100":
        raise RuntimeError("Only CIFAR-100 implemented in this demo.")

    root = Path("data") / "cifar100"
    root.mkdir(parents=True, exist_ok=True)
    _prepare_cifar(root, cfg)

    train_ds = torchvision.datasets.CIFAR100(
        root=root, train=True, download=False, transform=_build_transforms(cfg, True)
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=root, train=False, download=False, transform=_build_transforms(cfg, False)
    )

    num_tasks = cfg["dataset"]["num_tasks"]
    cpt = cfg["dataset"]["classes_per_task"]
    train_idx = _split_by_class(train_ds, num_tasks, cpt)
    test_idx = _split_by_class(test_ds, num_tasks, cpt)

    bs = cfg["global"]["batch_size"]
    g = torch.Generator().manual_seed(42)
    train_loaders, val_loaders, test_loaders = [], [], []
    for t in range(num_tasks):
        subset = Subset(train_ds, train_idx[t])
        n_val = max(1, int(0.1 * len(subset)))
        val_subset, train_subset = random_split(subset, [n_val, len(subset) - n_val], generator=g)
        train_loaders.append(
            DataLoader(
                train_subset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True
            )
        )
        val_loaders.append(
            DataLoader(
                val_subset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True
            )
        )
        test_loaders.append(
            DataLoader(
                Subset(test_ds, test_idx[t]),
                batch_size=bs,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        )
    return train_loaders, val_loaders, test_loaders
