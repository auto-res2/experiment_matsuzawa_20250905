"""src/preprocess.py
Dataset download, preprocessing and reproducibility helpers.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset

# -----------------------------------------------------------------------------
#                          REPRODUCIBILITY
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
#                             DATA  UTILS
# -----------------------------------------------------------------------------


def _as_path(p: str | Path) -> Path:
    """Helper that always returns a *Path* instance (idempotent)."""
    return p if isinstance(p, Path) else Path(p)


def _download_waterbirds(dataset_repo: str, cache_root: str | Path) -> None:
    """Trigger a *datasets* download so that subsequent calls are cache-hits."""
    cache_root = _as_path(cache_root)
    try:
        load_dataset(dataset_repo, cache_dir=str(cache_root / "hf_cache"))
    except Exception as exc:  # pragma: no cover – network issues are always possible
        raise RuntimeError(
            "Failed to download the Waterbirds dataset from the HuggingFace hub."
        ) from exc


class WaterbirdsTorch(torch.utils.data.Dataset):
    """Minimal PyTorch wrapper around the HuggingFace *Waterbirds* dataset."""

    def __init__(self, split: str, transform: T.Compose, cfg: dict):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train/validation/test")

        data_root = _as_path(cfg["data_root"])
        self.ds = load_dataset(
            cfg["dataset_repo"], split=split, cache_dir=str(data_root / "hf_cache")
        )
        self.transform = transform
        # expose column names once – avoids repetitive string look-ups
        self._lbl_key = "y" if "y" in self.ds.column_names else "label"
        self._place_key = "place"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = self.transform(item["image"])
        label = int(item[self._lbl_key])
        place = int(item[self._place_key])
        group_idx = label * 2 + place  # 0 … 3
        return img, label, group_idx

# -----------------------------------------------------------------------------
#                     DATALOADER  CONSTRUCTION
# -----------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Create *(train, val, test)* dataloaders according to *cfg*."""

    tf_train = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.67, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tf_eval = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = WaterbirdsTorch("train", tf_train, cfg)
    val_ds = WaterbirdsTorch("validation", tf_eval, cfg)
    test_ds = WaterbirdsTorch("test", tf_eval, cfg)

    def _make(ds, shuffle=False):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    return _make(train_ds, True), _make(val_ds), _make(test_ds)
