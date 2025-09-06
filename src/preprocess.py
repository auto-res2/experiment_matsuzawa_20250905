"""
preprocess.py – data download, verification and task stream builders
Only CIFAR-100 variants are needed for this iteration.
"""
from __future__ import annotations

import hashlib
import pathlib
import random
import tarfile
from typing import List, Tuple

import requests
import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from yaml import safe_load

# ---------------------------------------------------------------------------
DATA_ROOT = pathlib.Path("data")


# ---------------------------------------------------------------------------
#  Helper – checksums & downloads
# ---------------------------------------------------------------------------

def _sha1(path: pathlib.Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, tgt: pathlib.Path, checksum: str, kind: str = "md5") -> None:
    if tgt.exists():
        ok = (_md5(tgt) if kind == "md5" else _sha1(tgt)) == checksum
        if ok:
            return
    print(f"[Download] {url} -> {tgt}")
    tgt.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with tgt.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    ok = (_md5(tgt) if kind == "md5" else _sha1(tgt)) == checksum
    if not ok:
        raise RuntimeError("Checksum mismatch – aborting as per STRICT NO-FALLBACK rule")


# ---------------------------------------------------------------------------
#  CIFAR-100 continual variants
# ---------------------------------------------------------------------------

def _load_shared_cfg():
    return safe_load(pathlib.Path("config/config.yaml").read_text())


def build_split_cifar100(seed: int) -> Tuple[List[Subset], Dataset]:
    cfg = _load_shared_cfg()
    url, md5 = cfg["shared"]["cifar100_url"], cfg["shared"]["cifar100_md5"]

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    arc = DATA_ROOT / "cifar-100-python.tar.gz"
    _download(url, arc, md5, "md5")

    if not (DATA_ROOT / "cifar-100-python").exists():
        with tarfile.open(arc) as tf:
            tf.extractall(DATA_ROOT)

    tf_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            ),
        ]
    )
    tf_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    train_set = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=True, download=False, transform=tf_train
    )
    test_set = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=False, download=False, transform=tf_test
    )

    cls_order = list(range(100))
    random.Random(seed).shuffle(cls_order)
    tasks = [cls_order[i * 5 : (i + 1) * 5] for i in range(20)]  # 20 × 5-class tasks

    stream: List[Subset] = []
    for cls in tasks:
        idx = [i for i, y in enumerate(train_set.targets) if y in cls]  # type: ignore[attr-defined]
        stream.append(Subset(train_set, idx))

    return stream, test_set


# ---------------------------------------------------------------------------
#  100-task variant (one class per task)
# ---------------------------------------------------------------------------

def build_cifar100_one_class(seed: int):
    stream, test_set = build_split_cifar100(seed)

    single: List[Subset] = []
    for subset in stream:
        # which classes are inside this subset?
        labels = set(int(torchvision.datasets.CIFAR100.targets[i]) for i in subset.indices)  # type: ignore[attr-defined]
        for c in labels:
            idx = [i for i in subset.indices if torchvision.datasets.CIFAR100.targets[i] == c]  # type: ignore[attr-defined]
            single.append(Subset(subset.dataset, idx))

    single = single[:100]  # ensure 100 tasks exactly
    return single, test_set
