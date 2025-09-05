"""
preprocess.py – dataset download, task split and dataloader construction
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, random_split

DATA_DIR = Path("data")


# -----------------------------------------------------------------------------
#  Internal helpers
# -----------------------------------------------------------------------------

def _split_by_class(dataset, num_tasks: int, classes_per_task: int) -> List[List[int]]:
    """Return a list of indices per task (simple class // classes_per_task split)."""

    per_task: List[List[int]] = [[] for _ in range(num_tasks)]
    for idx, (_, y) in enumerate(dataset):
        task = y // classes_per_task
        if task < num_tasks:
            per_task[task].append(idx)
    return per_task


# -----------------------------------------------------------------------------
#  Public API – build continual data stream
# -----------------------------------------------------------------------------

def build_stream(cfg: Dict[str, int | str]) -> Tuple[List, List, List]:  # noqa: D401
    """Return (train_loaders, val_loaders, test_loaders) – one list entry per task."""

    name = str(cfg["name"]).lower()
    num_tasks = int(cfg["num_tasks"])
    cpt = int(cfg["classes_per_task"])
    batch_size = int(cfg.get("batch_size", 128))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if name == "cifar100":
        import torchvision.datasets as dset

        root = DATA_DIR / "cifar100"
        train_ds = dset.CIFAR100(
            root=root.as_posix(),
            train=True,
            download=True,
            transform=T.Compose(
                [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]
            ),
        )
        test_ds = dset.CIFAR100(
            root=root.as_posix(),
            train=False,
            download=True,
            transform=T.ToTensor(),
        )
    else:
        raise RuntimeError(f"Dataset {name} not supported in this demo refactor.")

    # ------------------------------------------------------------------
    train_idx = _split_by_class(train_ds, num_tasks, cpt)
    test_idx = _split_by_class(test_ds, num_tasks, cpt)

    train_loaders, val_loaders, test_loaders = [], [], []
    g = torch.Generator().manual_seed(42)

    for t in range(num_tasks):
        tr_subset = Subset(train_ds, train_idx[t])
        n = len(tr_subset)
        n_val = max(1, int(0.1 * n))
        val_subset, train_subset = random_split(tr_subset, [n_val, n - n_val], generator=g)

        train_loaders.append(
            DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        )
        val_loaders.append(
            DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        )
        test_loaders.append(
            DataLoader(
                Subset(test_ds, test_idx[t]),
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        )

    return train_loaders, val_loaders, test_loaders
