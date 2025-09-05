from __future__ import annotations
"""src/preprocess.py
Datasets and data-loading utilities (TorchVision + (optional) Avalanche).
If Avalanche is not available (or fails to install because of secondary
incompatible dependencies such as `proxsuite` + Python 3.11), we fall back
on a very light-weight re-implementation of the Split-CIFAR-100 benchmark
sufficient for the needs of this project.
"""
from typing import List, Tuple
import pathlib
import random

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as tvd

# -----------------------------------------------------------------------------
#                       Common transformations
# -----------------------------------------------------------------------------

_CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR_STD = (0.2675, 0.2565, 0.2761)


def _common_transforms(train: bool):
    aug: List[T.transforms] = []
    if train:
        aug += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    aug += [T.ToTensor(), T.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    return T.Compose(aug)


# ============================================================================
#  1)  Split-CIFAR-100 benchmark
# ============================================================================
# Attempt to import the official Avalanche implementation.  If that fails
# (e.g. because `avalanche-lib` cannot be installed on Python 3.11 due to the
# `proxsuite` wheel issue), we silently fall back on an in-house minimal
# counterpart that matches the subset of the API needed by the code-base.
# ============================================================================

try:
    from avalanche.benchmarks.classic import SplitCIFAR100 as _AvalancheSplitCIFAR100

    def get_split_cifar100(batch_size: int = 128, *, seed: int = 0):
        train_tf = _common_transforms(True)
        test_tf = _common_transforms(False)
        benchmark = _AvalancheSplitCIFAR100(
            n_experiences=20,
            seed=seed,
            return_task_id=True,
            train_transform=train_tf,
            eval_transform=test_tf,
        )
        return benchmark

except Exception:  # pragma: no cover – any error, not just ImportError

    # -----------------------------
    # Minimal re-implementation
    # -----------------------------

    class _Experience:
        """Light wrapper mimicking Avalanche's *experience* object."""

        def __init__(self, dataset: torch.utils.data.Dataset):
            self._dataset = dataset

        # The original Avalanche interface is `experience.dataset`, but in our
        # code-base we access it only through the `.dataloader` helper.
        def dataloader(self, *, batch_size: int = 128, num_workers: int = 0, shuffle: bool = True):
            return DataLoader(
                self._dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

    class _SimpleSplitCIFAR100Benchmark:
        """20-task Split-CIFAR-100 benchmark (5 classes per task).

        This covers just enough of Avalanche's API for the training loop used in
        `src/train.py` / `src/main.py` (i.e. `.train_stream` and
        `.test_stream[-1]`).
        """

        def __init__(self, *, batch_size: int = 128, seed: int = 0):
            self._batch_size = batch_size
            self._seed = seed
            self._build()

        # ---------------- private helpers ---------------------------------
        def _build(self):
            g = random.Random(self._seed)
            class_order = list(range(100))
            g.shuffle(class_order)

            # Load raw datasets ------------------------------------------------
            train_ds = tvd.CIFAR100(root="data", train=True, download=True)
            test_ds = tvd.CIFAR100(root="data", train=False, download=True)

            # Generate 20 experiences (5 classes each) ------------------------
            train_exps = []
            for i in range(0, 100, 5):
                cls_subset = set(class_order[i : i + 5])
                idxs = [j for j, y in enumerate(train_ds.targets) if y in cls_subset]
                subset = Subset(train_ds, idxs)
                subset.dataset.transform = _common_transforms(True)
                train_exps.append(_Experience(subset))

            # One single aggregated testing experience ------------------------
            test_ds.transform = _common_transforms(False)
            test_exp = _Experience(test_ds)

            self.train_stream = train_exps
            self.test_stream = [test_exp]

    # Public helper -----------------------------------------------------------

    def get_split_cifar100(batch_size: int = 128, *, seed: int = 0):
        """Return a light-weight Split-CIFAR-100 benchmark implementation."""
        return _SimpleSplitCIFAR100Benchmark(batch_size=batch_size, seed=seed)


# -----------------------------------------------------------------------------
#                       Streaming CIFAR loader (task-free)
# -----------------------------------------------------------------------------

def get_streaming_cifar(batch_size: int = 10):
    tf = _common_transforms(True)
    ds = tvd.CIFAR100(root="data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
