from __future__ import annotations
"""src/preprocess.py ––– dataset acquisition & preprocessing logic (patched)
Adds proper boolean masks for **ogbn-arxiv** so that downstream training code
can reuse the generic Planetoid‐style train/val/test handling.
"""
import warnings
from pathlib import Path
from typing import Dict, Callable

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor
from torch_geometric.transforms import NormalizeFeatures

try:
    from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore
except ImportError:  # pragma: no cover – optional dependency
    warnings.warn("'ogb' not found – ogbn_* datasets will be unavailable.  Install via `pip install ogb`. ")
    PygNodePropPredDataset = None  # type: ignore

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Dataset factory registry
# ----------------------------------------------------------------------------

def _wrap_ogbn_arxiv() -> object:  # returns a *dataset* compatible with Planetoid API
    if PygNodePropPredDataset is None:
        raise RuntimeError("Requested ogbn-arxiv but 'ogb' package is not installed.")

    pyg_dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(_DATA_ROOT))
    split_idx = pyg_dataset.get_idx_split()
    data: Data = pyg_dataset[0]

    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[split_idx["train"]] = True
    val_mask[split_idx["valid"]] = True
    test_mask[split_idx["test"]] = True

    # Attach masks directly – this mutates the underlying Data object inside
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return pyg_dataset  # return the *dataset* (list-like) to preserve [0] indexing


DATASET_CLASS_MAP: Dict[str, Callable[[], object]] = {
    "Cora":           lambda: Planetoid(str(_DATA_ROOT), "Cora", transform=NormalizeFeatures()),
    "Pubmed":         lambda: Planetoid(str(_DATA_ROOT), "Pubmed", transform=NormalizeFeatures()),
    "Chameleon":      lambda: WikipediaNetwork(str(_DATA_ROOT), "chameleon", transform=NormalizeFeatures()),
    "Actor":          lambda: Actor(str(_DATA_ROOT), transform=NormalizeFeatures()),
}

if PygNodePropPredDataset is not None:
    DATASET_CLASS_MAP["ogbn_arxiv"] = _wrap_ogbn_arxiv

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def get_dataset(name: str):
    """Return a **PyG dataset object** for the requested name."""
    if name not in DATASET_CLASS_MAP:
        raise ValueError(f"Unknown dataset '{name}'.  Available: {list(DATASET_CLASS_MAP)}")
    return DATASET_CLASS_MAP[name]()
