from __future__ import annotations
"""src/preprocess.py ––– dataset acquisition & preprocessing logic"""
import warnings
from pathlib import Path
from typing import Dict, Callable

import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor
from torch_geometric.transforms import NormalizeFeatures

try:
    from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore
except ImportError:
    warnings.warn("'ogb' not found – ogbn_* datasets will be unavailable.  Install via `pip install ogb`. ")
    PygNodePropPredDataset = None  # type: ignore

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_DATA_ROOT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Dataset factory registry
# ----------------------------------------------------------------------------

DATASET_CLASS_MAP: Dict[str, Callable[[], object]] = {
    "Cora":           lambda: Planetoid(str(_DATA_ROOT), "Cora", transform=NormalizeFeatures()),
    "Pubmed":         lambda: Planetoid(str(_DATA_ROOT), "Pubmed", transform=NormalizeFeatures()),
    "Chameleon":      lambda: WikipediaNetwork(str(_DATA_ROOT), "chameleon", transform=NormalizeFeatures()),
    "Actor":          lambda: Actor(str(_DATA_ROOT), transform=NormalizeFeatures()),
}

if PygNodePropPredDataset is not None:
    DATASET_CLASS_MAP["ogbn_arxiv"] = lambda: PygNodePropPredDataset(name="ogbn-arxiv", root=str(_DATA_ROOT))

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def get_dataset(name: str):
    """Return a **PyG dataset object** for the requested name."""
    if name not in DATASET_CLASS_MAP:
        raise ValueError(f"Unknown dataset '{name}'.  Available: {list(DATASET_CLASS_MAP)}")
    return DATASET_CLASS_MAP[name]()
