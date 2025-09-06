"""
preprocess.py – data loading, synthetic datasets, curvature estimation & misc utilities
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Tuple

import torch
import networkx as nx
from torch_geometric.datasets import (
    Amazon,
    Coauthor,
    Planetoid,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_self_loops, to_networkx
from torch_geometric.data import Data
# The OGB dependency is optional – we import lazily so the codebase can still
# execute in environments without the heavy OGB package.
try:
    from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore
except ImportError:  # pragma: no cover – light-weight fallback
    PygNodePropPredDataset = None  # type: ignore

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  Reproducibility helper
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
#  Dataset loaders
# -----------------------------------------------------------------------------

def _load_planetoid(name: str):
    """Return the canonical Planetoid dataset object.

    PyG expects the following exact names: "Cora", "CiteSeer", "PubMed".  We map
    a variety of user-supplied casings to these canonical identifiers.
    """
    _canonical = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
    }
    if name.lower() not in _canonical:
        raise RuntimeError(f"Unknown Planetoid dataset key: {name}")
    ds = Planetoid(root=DATA_ROOT / _canonical[name.lower()], name=_canonical[name.lower()],
                   transform=NormalizeFeatures())
    return ds[0]


def _load_wiki(name: str):
    # WikipediaNetwork internally uses lower-case identifiers ("chameleon", …)
    ds = WikipediaNetwork(root=DATA_ROOT / name, name=name, transform=NormalizeFeatures())
    return ds[0]


def _load_webkb(name: str):
    # WebKB expects capitalised names
    _canonical = {"texas": "Texas", "cornell": "Cornell", "wisconsin": "Wisconsin"}
    name_c = _canonical[name.lower()]
    ds = WebKB(root=DATA_ROOT / name_c, name=name_c, transform=NormalizeFeatures())
    return ds[0]


def _load_ogb(name: str):
    if PygNodePropPredDataset is None:
        raise ImportError("OGB is not installed – add 'ogb' to your dependencies if you "
                          "need OGB datasets.")
    ds = PygNodePropPredDataset(name=name, root=DATA_ROOT / name)
    data = ds[0]
    split = ds.get_idx_split()

    n = data.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split["train"]] = True
    val_mask[split["valid"]] = True
    test_mask[split["test"]] = True
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
    return data


# -----------------------------------------------------------------------------
#  Synthetic RingTransfer benchmark
# -----------------------------------------------------------------------------

def _build_ringtransfer(
    num_nodes: int = 30_000, p_long: float = 0.01, num_classes: int = 10
):
    g = nx.cycle_graph(num_nodes)
    for i in range(num_nodes):
        if random.random() < p_long:
            g.add_edge(i, (i + num_nodes // 2) % num_nodes)

    x_feat = []
    for i in range(num_nodes):
        deg = g.degree[i]
        x_feat.append([deg == 2, deg == 3, math.sin(i / 1000.0), math.cos(i / 1000.0)])
    x = torch.tensor(x_feat, dtype=torch.float32)
    y = torch.tensor(
        [(i * num_classes) // num_nodes for i in range(num_nodes)], dtype=torch.long
    )

    src, dst = zip(*g.edges())  # tuples -> two lists
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # make undirected explicit
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data = Data(x=x, edge_index=edge_index, y=y)
    n = num_nodes
    idx = torch.randperm(n)
    tr, va = int(0.6 * n), int(0.8 * n)
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    data.train_mask[idx[:tr]] = True
    data.val_mask[idx[tr:va]] = True
    data.test_mask[idx[va:]] = True
    return data


# -----------------------------------------------------------------------------
#  Public interface
# -----------------------------------------------------------------------------

def load_dataset(name: str):
    name_l = name.lower()
    if name_l in {"cora", "citeseer", "pubmed"}:
        return _load_planetoid(name_l)
    if name_l in {"chameleon", "squirrel"}:
        return _load_wiki(name_l)
    if name_l in {"texas", "cornell", "wisconsin"}:
        return _load_webkb(name_l)
    if name_l in {"ogbn-products", "ogbn-arxiv"}:
        return _load_ogb(name_l)
    if name_l == "ringtransfer":
        return _build_ringtransfer()
    raise RuntimeError(f"Dataset {name} not supported.")


# -----------------------------------------------------------------------------
#  Ollivier-Ricci curvature approximation helper
# -----------------------------------------------------------------------------

def compute_curvature(
    data, proj_dim: int = 32, eps: float = 0.1, device: str = "cpu"
):
    """Return (edge_curvatures, node_mean_curvatures).

    GraphRicciCurvature (and its heavy Networkit dependency) often lacks wheels
    for the newest Python versions.  We therefore *attempt* to compute true
    Ollivier-Ricci curvature, but fall back to a cheap zero-curvature heuristic
    if the library is unavailable.
    """
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci  # type: ignore

        g_nx = to_networkx(data, to_undirected=True)
        orc = OllivierRicci(
            g_nx, method="Sinkhorn", alpha=0, dimension=proj_dim, sinkhorn_epsilon=eps
        )
        orc.compute_ricci_curvature()
        curv_dict = {
            tuple(sorted((u, v))): d["ricciCurvature"] for u, v, d in g_nx.edges(data=True)
        }

        src, dst = data.edge_index.cpu().numpy()
        kappa_edge = torch.tensor(
            [curv_dict[tuple(sorted((int(u), int(v))))] for u, v in zip(src, dst)],
            dtype=torch.float16,
            device=device,
        ).clamp(-3, 3)

        # node-level average
        kappa_node = torch.zeros(data.num_nodes, dtype=torch.float32, device=device)
        kappa_node.index_add_(0, torch.tensor(src, device=device), kappa_edge.float())
        kappa_node.index_add_(0, torch.tensor(dst, device=device), kappa_edge.float())
        deg = torch.bincount(torch.tensor(src, device=device), minlength=data.num_nodes).clamp(
            min=1
        )
        kappa_node = (kappa_node / deg).clamp(-3, 3)
        return kappa_edge, kappa_node
    except Exception as err:  # pragma: no cover – fallback branch
        # We purposefully *do not* crash the whole experiment here – curvature is
        # an important but *optional* signal.  A warning is printed so the user
        # knows what happened.
        print(
            "[WARN] GraphRicciCurvature unavailable or failed (" + str(err) + ") – "
            "using zero-curvature fallback."
        )
        kappa_edge = torch.zeros(
            data.edge_index.size(1), dtype=torch.float16, device=device, requires_grad=False
        )
        kappa_node = torch.zeros(
            data.num_nodes, dtype=torch.float32, device=device, requires_grad=False
        )
        return kappa_edge, kappa_node
