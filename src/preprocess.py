"""src/preprocess.py
Data loading & preprocessing utilities.  All dataset-specific logic is isolated
here so the rest of the codebase remains cleaner.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    from wilds import get_dataset
except ImportError:
    get_dataset = None  # Wilds is optional – only used for Waterbirds/CelebA

LOGGER = logging.getLogger("pcd.preprocess")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# UTILS ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# DATASET FACTORY --------------------------------------------------------------
# -----------------------------------------------------------------------------

class DatasetFactory:
    """Factory that returns (train, val, test, meta) tuples for supported data."""

    @staticmethod
    def cifar10() -> Tuple[Any, Any, Any, Dict]:
        tf_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandAugment(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )
        tf_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )
        train_full = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=tf_train)
        test = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=tf_test)
        n_val = int(0.1 * len(train_full))
        train, val = random_split(train_full, [len(train_full) - n_val, n_val])
        return train, val, test, {"num_classes": 10}

    @staticmethod
    def waterbirds():
        if get_dataset is None:
            raise RuntimeError("'wilds' not installed – cannot load Waterbirds.")
        ds = get_dataset("waterbirds", root_dir=str(DATA_DIR), download=True)
        tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        ds.transform = tf
        return ds.get_subset("train"), ds.get_subset("val"), ds.get_subset("test"), {
            "num_classes": 2,
            "group_labels_present": True,
        }

    @staticmethod
    def celeba():
        if get_dataset is None:
            raise RuntimeError("'wilds' not installed – cannot load CelebA.")
        ds = get_dataset("celebA", root_dir=str(DATA_DIR), download=True)
        tf = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        ds.transform = tf
        return ds.get_subset("train"), ds.get_subset("val"), ds.get_subset("test"), {
            "num_classes": 2,
            "group_labels_present": True,
        }


# -----------------------------------------------------------------------------
# DATA LOADER CONVENIENCE ------------------------------------------------------
# -----------------------------------------------------------------------------

def make_loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


__all__ = ["set_seed", "DatasetFactory", "make_loader"]