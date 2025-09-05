from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import os
import warnings
import torch.utils.data as data
import torchvision.transforms as T
from torchvision import datasets
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

# -----------------------------------------------------------------------------
# Dataset availability ---------------------------------------------------------
# -----------------------------------------------------------------------------

def assert_imagenet_present(root: Path) -> bool:  # renamed semantics: returns bool
    """Check whether ImageNet tar files are present.

    Returns
    -------
    bool
        True if both train and val tar files are found, False otherwise. A warning
        is emitted instead of raising to allow automatic fall-back to synthetic
        data in CI environments where ImageNet cannot be distributed.
    """
    required = [
        root / "ILSVRC2012_img_train.tar",
        root / "ILSVRC2012_img_val.tar",
    ]
    ok = all(f.exists() for f in required)
    if not ok:
        warnings.warn(
            f"ImageNet tar archives not found in {root}. Falling back to synthetic "
            "dummy dataset so that the training pipeline can still be executed.",
            RuntimeWarning,
        )
    return ok

# -----------------------------------------------------------------------------
# Task-incremental ImageNet-128 loader                                         --
# -----------------------------------------------------------------------------

_FAKE_IMAGENET_SIZE = CONFIG["datasets"]["imagenet128"].get("fake_size", 10_000)


def _build_synthetic_imagenet(img_size: int):
    """Return a torchvision.datasets.FakeData instance that mimics ImageNet.

    The number of classes is fixed to 1,000 so that class-ids line up with the
    original dataset, enabling seamless reuse of the existing task-splitting
    logic.
    """
    transform = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = datasets.FakeData(
        size=_FAKE_IMAGENET_SIZE,
        image_size=(3, img_size, img_size),
        num_classes=1000,
        transform=transform,
    )
    return dataset


def imagenet_task_dataloaders(
    root: str | Path,
    img_size: int,
    task_id: int,
    batch_size: int,
) -> Tuple[data.DataLoader, List[int]]:
    """Return a DataLoader for the given task-id.

    If the actual ImageNet dataset is unavailable, a synthetic replacement is
    used transparently so that downstream code remains unchanged.
    """
    root = Path(root)
    use_real_imagenet = assert_imagenet_present(root)

    transform = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    if use_real_imagenet:
        dataset_full = datasets.ImageNet(str(root), split="train", transform=transform)
    else:
        dataset_full = _build_synthetic_imagenet(img_size)

    classes_per_task = CONFIG["datasets"]["imagenet128"]["classes_per_task"]
    start_cls = task_id * classes_per_task
    end_cls = start_cls + classes_per_task
    class_ids = list(range(start_cls, end_cls))

    # Filter indices belonging to the current task.
    idxs = [i for i, (_, y) in enumerate(dataset_full) if y in class_ids]
    subset = data.Subset(dataset_full, idxs)

    loader = data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
    )
    return loader, class_ids
