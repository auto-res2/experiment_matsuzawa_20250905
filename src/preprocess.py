"""src/preprocess.py
--------------------------------------------------------------------
Dataset loading, preprocessing utilities and seed control.
"""
from __future__ import annotations

# ----------------- standard lib ----------------------------------
import os
import random
from pathlib import Path
from typing import Tuple

# ----------------- third-party -----------------------------------
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import to_undirected


# -----------------------------------------------------------------
#  Reproducibility helpers                                         
# -----------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# -----------------------------------------------------------------
#  Loaders                                                         
# -----------------------------------------------------------------

def _boolean_masks(data):
    data.train_mask = data.train_mask.bool()
    data.val_mask = data.val_mask.bool()
    data.test_mask = data.test_mask.bool()
    return data


def load_planetoid(name: str, root: Path):
    ds = Planetoid(root=str(root / name), name=name)
    data = ds[0]
    data.edge_index = to_undirected(data.edge_index)
    return _boolean_masks(data)


def load_chameleon(root: Path):
    ds = WikipediaNetwork(
        root=str(root / "Chameleon"), name="chameleon", geom_gcn_preprocess=False, split="random"
    )
    data = ds[0]
    data.edge_index = to_undirected(data.edge_index)

    # create Planetoid-style boolean masks (75/100/rest per class)
    num_classes = int(data.y.max().item() + 1)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros_like(data.train_mask)
    data.test_mask = torch.zeros_like(data.train_mask)

    for c in range(num_classes):
        idx = torch.where(data.y == c)[0]
        idx = idx[torch.randperm(idx.size(0))]
        data.train_mask[idx[:75]] = True
        data.val_mask[idx[75:175]] = True
        data.test_mask[idx[175:]] = True
    return data


def load_ogbn_arxiv(root: Path):
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OGB package not installed – install ogb>=1.3.6") from exc

    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "ogbn-arxiv"))
    data = ds[0]
    split_idx = ds.get_idx_split()
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[split_idx["train"]] = True
    val_mask[split_idx["valid"]] = True
    test_mask[split_idx["test"]] = True
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    data.edge_index = to_undirected(data.edge_index)
    return data
