"""
preprocess.py
=============
Dataset download / generation and generic preprocessing utilities live here.
All datasets are stored below `data/` to keep the project root tidy.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import torch
from scipy import sparse as sp
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from torch_geometric.utils import (
    add_self_loops,
    from_scipy_sparse_matrix,
)
from ogb.nodeproppred import PygNodePropPredDataset

__all__ = [
    "get_planetoid",
    "get_chameleon",
    "get_webkb",
    "get_ogbn_arxiv",
    "make_sbm_graph",
]

_DATA_DIR = Path("data")
_DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  Real-world benchmark datasets
# ---------------------------------------------------------------------------

def get_planetoid(name: str):
    return Planetoid(root=_DATA_DIR / name, name=name)[0]


def get_chameleon():
    return WikipediaNetwork(root=_DATA_DIR / "Chameleon", name="chameleon")[0]


def get_webkb(name: str):
    return WebKB(root=_DATA_DIR / name, name=name)[0]


def get_ogbn_arxiv():
    ds = PygNodePropPredDataset(root=_DATA_DIR / "ogbn_arxiv", name="ogbn-arxiv")
    data = ds[0]
    split_idx = ds.get_idx_split()
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros_like(data.train_mask)
    data.test_mask = torch.zeros_like(data.train_mask)
    data.train_mask[split_idx["train"]] = True
    data.val_mask[split_idx["valid"]] = True
    data.test_mask[split_idx["test"]] = True
    return data

# ---------------------------------------------------------------------------
# 2.  Synthetic Stochastic Block Model generator (Exp-1)
# ---------------------------------------------------------------------------

def make_sbm_graph(
    n_per_block: int = 1000,
    h: float = 0.7,
    sigma: float = 0.5,
    d: int = 16,
    seed: int = 0,
):
    rng = np.random.RandomState(seed)
    sizes = [n_per_block, n_per_block]
    p_in, p_out = h, 1.0 - h
    probs = [[p_in, p_out], [p_out, p_in]]

    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    A: sp.csr_matrix = nx.to_scipy_sparse_matrix(g, format="csr")

    attrs = rng.randn(2 * n_per_block, d).astype("float32")
    attrs[n_per_block:] += sigma * rng.randn(n_per_block, d).astype("float32")
    labels = np.concatenate([np.zeros(n_per_block), np.ones(n_per_block)]).astype("int64")

    edge_index, _ = from_scipy_sparse_matrix(A)
    edge_index, _ = add_self_loops(edge_index)

    data = Data(x=torch.from_numpy(attrs), edge_index=edge_index, y=torch.from_numpy(labels))

    # -------- create 60/20/20 train/val/test split (class-balanced) ------
    y_np = data.y.numpy()
    idx = np.arange(len(y_np))
    train_mask = np.zeros(len(y_np), dtype=bool)
    val_mask = np.zeros_like(train_mask)
    test_mask = np.zeros_like(train_mask)
    for cls in [0, 1]:
        cls_idx = idx[y_np == cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        train_mask[cls_idx[: int(0.6 * n)]] = True
        val_mask[cls_idx[int(0.6 * n) : int(0.8 * n)]] = True
        test_mask[cls_idx[int(0.8 * n) :]] = True

    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)
    return data
