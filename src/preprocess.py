"""
preprocess.py – data loading, perturbations, configuration utilities
"""
from __future__ import annotations

import random
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import add_self_loops, get_laplacian, to_undirected
import torch_geometric

# robust-attack imports (optional)
try:
    from deeprobust.graph.global_attack import Metattack
except Exception:
    Metattack = None  # defer error until actually used

_CONFIG_PATH = Path("config/config.yaml")


def get_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError("config/config.yaml not found. Have you created it?")
    return yaml.safe_load(_CONFIG_PATH.read_text())


CFG = get_config()

# ----------------------------------------------------------------------------
#  Reproducibility helper
# ----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------
#  Dataset loading & caching
# ----------------------------------------------------------------------------

DATA_DIR = Path("data").absolute()
CACHE_DIR = DATA_DIR / "processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def load_dataset(name: str) -> Data:
    """Load a dataset and augment it with degree, curvature and Laplacian data.

    The Ollivier–Ricci curvature computation is cached to speed-up subsequent runs.
    """
    name_l = name.lower()
    root = DATA_DIR / name_l
    if name_l in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(root=str(root), name=name.capitalize())
        data = ds[0]
    elif name_l in {"chameleon", "squirrel"}:
        ds = WikipediaNetwork(root=str(root), name=name_l, geom_gcn_preprocess=True)
        data = ds[0]
    else:
        raise RuntimeError(f"Unknown dataset {name}")

    # feature normalisation --------------------------------------------------
    data.x = data.x / (data.x.sum(1, keepdim=True) + 1e-12)

    # make the graph undirected and add self-loops ---------------------------
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    # degree -----------------------------------------------------------------
    deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes).float()
    data.deg = deg

    # Ollivier–Ricci curvature ----------------------------------------------
    curv_file = CACHE_DIR / f"{name_l}_edge_curv.npy"
    if curv_file.exists():
        edge_curv = np.load(curv_file)
        edges_ordered = np.load(curv_file.with_suffix("_edges.npy"))
    else:
        print(f"Computing Ollivier–Ricci curvature for {name} (one-off)…")
        g = nx.Graph()
        # ensure *consistent* Python-int node types throughout ----------------
        g.add_nodes_from(range(int(data.num_nodes)))
        # convert edge index to a list of Python-int tuples (avoids np.int64 keys)
        edges_ordered = [tuple(map(int, e)) for e in data.edge_index.t().tolist()]
        g.add_edges_from(edges_ordered)

        # run curvature computation
        orc = OllivierRicci(g, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()

        # curvature for each edge (same order as *edges_ordered*)
        edge_curv = np.array([orc.G[u][v]["ricciCurvature"] for u, v in edges_ordered])
        # cache both curvature values and the corresponding edge list so that
        # ordering is reproduced exactly the next time we load from disk.
        np.save(curv_file, edge_curv)
        np.save(curv_file.with_suffix("_edges.npy"), np.asarray(edges_ordered, dtype=np.int64))

    data.edge_curv = torch.tensor(edge_curv, dtype=torch.float)

    # node-level curvature: mean of incident edge curvatures ------------------
    node_curv = torch.zeros(data.num_nodes, dtype=torch.float)
    for (u, v), c in zip(edges_ordered, edge_curv):
        node_curv[u] += c
        node_curv[v] += c
    deg_clamped = deg.clone()
    deg_clamped[deg_clamped == 0] = 1.0
    data.node_curv = node_curv / deg_clamped

    # Laplacian (normalised) -------------------------------------------------
    ei, ew = get_laplacian(data.edge_index, normalization="sym")
    data.lap_edge_index, data.lap_edge_weight = ei, ew

    return data


# ----------------------------------------------------------------------------
#  Robustness helpers – topology & feature perturbations
# ----------------------------------------------------------------------------

def apply_meta_attack(data: Data, perturb_ratio: float, seed: int) -> Data:
    if Metattack is None:
        raise RuntimeError("deeprobust is required for MetaAttack.")
    set_seed(seed)
    adj = torch_geometric.utils.to_scipy_sparse_matrix(
        data.edge_index, num_nodes=data.num_nodes
    )
    features = data.x.numpy()
    labels = data.y.numpy()
    attacker = Metattack(
        model=None,
        nnodes=data.num_nodes,
        feature_shape=features.shape,
        attack_structure=True,
        attack_features=False,
        device="cpu",
    )
    idx_train = data.train_mask.nonzero(as_tuple=True)[0].numpy()
    attacker.attack(
        adj,
        features,
        labels,
        idx_train,
        n_perturbations=int(perturb_ratio * (adj.nnz // 2)),
    )
    modified_adj = attacker.modified_adj
    edge_idx_ptb = torch_geometric.utils.from_scipy_sparse_matrix(modified_adj)[0]
    data_ptb = data.clone()
    data_ptb.edge_index = edge_idx_ptb
    return data_ptb


def mask_features(data: Data, mask_ratio: float, seed: int) -> Data:
    set_seed(seed)
    x = data.x.clone()
    nnz = (x != 0).nonzero(as_tuple=False)
    idx = torch.randperm(nnz.size(0))[: int(mask_ratio * nnz.size(0))]
    x[nnz[idx][:, 0], nnz[idx][:, 1]] = 0.0
    data_masked = data.clone()
    data_masked.x = x
    return data_masked
