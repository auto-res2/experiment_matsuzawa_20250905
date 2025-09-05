"""src.preprocess – data loading & dataset handling extracted from the original script."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

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

def assert_imagenet_present(root: Path) -> None:
    required = [
        root / "ILSVRC2012_img_train.tar",
        root / "ILSVRC2012_img_val.tar",
    ]
    if not all(f.exists() for f in required):
        raise FileNotFoundError(
            f"ImageNet tar archives not found in {root}. "
            "Please download them manually from https://www.image-net.org/download "
            "and place them under the given directory."
        )

# -----------------------------------------------------------------------------
# Task-incremental ImageNet-128 loader                                         --
# -----------------------------------------------------------------------------

def imagenet_task_dataloaders(
    root: str | Path,
    img_size: int,
    task_id: int,
    batch_size: int,
) -> Tuple[data.DataLoader, List[int]]:
    transform = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset_full = datasets.ImageNet(str(root), split="train", transform=transform)

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
