"""src/preprocess.py – data downloading & continual-learning task split"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Tuple

import torch
import torch.utils.data as td
import torchvision

__all__ = [
    "get_data_stream",
]


def _build_cifar100_stream(
    num_tasks: int,
    classes_per_task: int,
    root: Path,
) -> Tuple[List[td.Dataset], List[td.Dataset]]:
    """Download CIFAR-100 (if necessary) and split it into sequential tasks."""

    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True)

    # ------------------------------------------------------------------
    #  Create the class-incremental split
    # ------------------------------------------------------------------
    assert num_tasks * classes_per_task == 100, "Invalid task split for CIFAR-100"

    permuted_classes = list(range(100))  # could be randomised → reproducibility
    task_cls = [
        permuted_classes[i * classes_per_task : (i + 1) * classes_per_task]
        for i in range(num_tasks)
    ]

    def _subset(dataset, cls_subset):
        idx = [j for j, (_, y) in enumerate(dataset) if y in cls_subset]
        imgs = torch.stack([dataset[j][0] for j in idx])
        labels = torch.tensor([dataset[j][1] for j in idx])
        return td.TensorDataset(imgs, labels)

    train_stream = [_subset(train_set, c) for c in task_cls]
    test_stream = [_subset(test_set, c) for c in task_cls]

    return train_stream, test_stream


# -----------------------------------------------------------------------------
#  Public entry point – other modules only import this one function
# -----------------------------------------------------------------------------

def get_data_stream(dataset_cfg: dict) -> Tuple[List[td.Dataset], List[td.Dataset]]:
    """Return (train_stream, test_stream) according to *dataset_cfg* dict."""

    name = dataset_cfg["name"].lower()
    if name == "cifar100":
        return _build_cifar100_stream(
            num_tasks=dataset_cfg["num_tasks"],
            classes_per_task=dataset_cfg["classes_per_task"],
            root=Path("data") / "cifar100",
        )

    # ------------------------------------------------------------------
    #  For datasets that cannot be downloaded automatically we raise a
    #  descriptive error so the user can fix it instead of silently failing.
    # ------------------------------------------------------------------
    raise RuntimeError(
        textwrap.dedent(
            f"""
            Dataset '{dataset_cfg['name']}' requires manual download or special
            credentials that are not available inside this execution sandbox.
            Please place the dataset at 'data/{dataset_cfg['name']}/' and rerun.
            """
        )
    )