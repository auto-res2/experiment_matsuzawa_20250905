from __future__ import annotations

"""src/preprocess.py – downloading, data transforms & random seed utility.
    NOTE:   Added `repo_type='dataset'` when calling snapshot_download to fix
            401 errors caused by defaulting to model repos.
"""

import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T

# ---------------------------------------------------------------------------
#                           Reproducibility helper
# ---------------------------------------------------------------------------

def set_seed(seed: int = 0):  # noqa: D401 simple name OK
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#                           torchvision transforms
# ---------------------------------------------------------------------------

def transforms_224(train: bool = True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
#                        HuggingFace dataset download helper
# ---------------------------------------------------------------------------


def _hash_repo(repo: str) -> str:
    return hashlib.sha1(repo.encode()).hexdigest()[:8]


def download_hf_dataset(repo: str, *, data_root: str | Path = "data") -> Path:
    """Download a public **dataset** repository from the 🤗 Hub using the cache.

    The hub differentiates between *model* and *dataset* repos.  The original
    implementation omitted `repo_type='dataset'`, causing a 401 / 404 error
    for legitimate dataset IDs.  This patch fixes that oversight and keeps the
    rest of the semantics identical.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover – missing dependency
        raise RuntimeError("huggingface_hub not installed – cannot fetch dataset") from exc

    data_root = Path(data_root)
    cache_dir = data_root / f"hf_{_hash_repo(repo)}"
    if cache_dir.exists():
        return cache_dir

    try:
        path = Path(
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",  # <-- crucial fix
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
        )
    except Exception as exc:
        # Clean partial downloads so that re-tries start from scratch --------
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to download {repo}: {exc}") from exc
    return path