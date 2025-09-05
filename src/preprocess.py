"""
src/preprocess.py – dataset & data-loader helpers
Only CIFAR-100 is supported out-of-the-box because it is readily
available through torchvision and small enough for quick CI runs.  The
function returns three streams (lists) of DataLoader objects – one list
per task – that main.py zips together during the continual-learning run.

If you add new datasets make sure to implement task/class partitioning
otherwise the rest of the pipeline will stay unchanged.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# Generic helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------


def _split_by_class(
    dataset: datasets.CIFAR100, *, num_tasks: int, classes_per_task: int
) -> List[List[int]]:
    """Return *indices* for each task such that classes do not overlap."""

    # CIFAR-100 classes are ordered (0-99); we simply slice them.
    class_idxs: List[List[int]] = [[] for _ in range(num_tasks)]
    for idx, (_, target) in enumerate(dataset):
        task_id = target // classes_per_task
        if task_id < num_tasks:  # guard against rounding errors
            class_idxs[task_id].append(idx)
    return class_idxs


# -----------------------------------------------------------------------------
# Public API -------------------------------------------------------------------
# -----------------------------------------------------------------------------


def build_stream(dataset_cfg: Dict) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    """Create class-incremental data loaders for train/val/test.

    The current implementation supports *only* CIFAR-100 because that is
    what the default experiment requires.  Attempting to use another
    dataset will raise an informative error – add your own dataset logic
    where indicated.
    """

    name = dataset_cfg.get("name", "").lower()
    if name != "cifar100":
        raise ValueError(
            f"Unsupported dataset '{name}'.  Only 'cifar100' is implemented in the OSS stub."
        )

    num_tasks: int = dataset_cfg["num_tasks"]
    classes_per_task: int = dataset_cfg["classes_per_task"]
    batch_size: int = dataset_cfg.get("batch_size", 128)  # fallback

    # ------------------------------------------------------------------
    # Download / load CIFAR-100 ----------------------------------------
    # ------------------------------------------------------------------
    root = Path("data/cifar100")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )

    train_ds = datasets.CIFAR100(root=str(root), train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100(root=str(root), train=False, download=True, transform=transform)

    # ------------------------------------------------------------------
    # Build per-task subsets -------------------------------------------
    # ------------------------------------------------------------------
    train_indices_per_task = _split_by_class(train_ds, num_tasks=num_tasks, classes_per_task=classes_per_task)
    test_indices_per_task = _split_by_class(test_ds, num_tasks=num_tasks, classes_per_task=classes_per_task)

    train_stream: List[DataLoader] = []
    val_stream: List[DataLoader] = []
    test_stream: List[DataLoader] = []

    for task_id in range(num_tasks):
        tr_subset = Subset(train_ds, train_indices_per_task[task_id])
        te_subset = Subset(test_ds, test_indices_per_task[task_id])

        # Naïve split: 90% train / 10% val
        n_tr = len(tr_subset)
        n_val = max(1, math.floor(0.1 * n_tr))
        val_subset, tr_subset = torch.utils.data.random_split(
            tr_subset, [n_val, n_tr - n_val], generator=torch.Generator().manual_seed(42)
        )

        train_stream.append(
            DataLoader(tr_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        )
        val_stream.append(
            DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        )
        test_stream.append(
            DataLoader(te_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        )

    return train_stream, val_stream, test_stream
