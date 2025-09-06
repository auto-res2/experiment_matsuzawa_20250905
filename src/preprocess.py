from __future__ import annotations

"""
preprocess.py – data downloading, basic image transforms, and reproducible
random-seed setup.  These helpers are intentionally lightweight so they
can be imported by *both* the training and evaluation scripts without
creating circular dependencies.
"""

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torchvision import transforms as T

# ---------------------------------------------------------------------------
#                           SEED  &  DETERMINISM
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Make a best-effort attempt at fully deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # `benchmark=True` gives faster but still deterministic kernels for fixed
    # input shapes.  `deterministic=True` would be *fully* reproducible but
    # unreasonably slow for conv-heavy workloads.  Our experiments chiefly use
    # ViTs so this compromise is acceptable.
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
#                           DATA    HELPERS
# ---------------------------------------------------------------------------

def download_hf_dataset(repo: str, *, data_root: str) -> Path:
    """Download a HuggingFace *dataset* snapshot into *data_root*.

    The huggingface-hub helper defaults to repo_type="model" which causes a
    confusing 401/404 when the repository is in the *datasets* namespace.  We
    therefore pass `repo_type="dataset"` explicitly.
    """
    from huggingface_hub import snapshot_download

    tgt_dir = Path(data_root) / repo.replace("/", "__")
    if tgt_dir.exists():
        return tgt_dir

    print(f"[Info ] Downloading dataset {repo} …")
    tgt_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",  # critical fix – avoid unauthorised / not-found errors
        local_dir=tgt_dir,
        local_dir_use_symlinks=False,
    )
    return tgt_dir


# ---------------------------------------------------------------------------
#                     STANDARD  224×224  IMAGE  TRANSFORMS
# ---------------------------------------------------------------------------

def transforms_224(*, train: bool = True):
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.RandAugment(num_ops=2, magnitude=10),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )