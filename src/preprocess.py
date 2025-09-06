import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, Dict

import networkx as nx
import torch
import yaml
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.datasets import Amazon, Coauthor
from torch_geometric.transforms import NormalizeFeatures
from ogb.nodeproppred import PygNodePropPredDataset

# ---------------------------------------------------------------------------
#  Light-weight utilities (kept local to avoid separate file)
# ---------------------------------------------------------------------------

@contextmanager
def timer(msg: str):
    import time
    t = time.time()
    yield
    print(f"{msg} – {time.time() - t:.2f}s", flush=True)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Dataset utilities
# ---------------------------------------------------------------------------

DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(exist_ok=True)


def _load_planetoid(name: str):
    return Planetoid(root=DATA_DIR / name, name=name, transform=NormalizeFeatures())[0]


def _load_ogb(name: str):
    dataset = PygNodePropPredDataset(name, root=DATA_DIR / name)
    return dataset[0], dataset.get_idx_split()


def _load_wikipedia(name: str):
    return WikipediaNetwork(root=DATA_DIR / name, name=name, transform=NormalizeFeatures())[0]


def get_dataset(name: str):
    """Returns a torch_geometric.data.Data object (single graph)."""
    with timer(f"Loading dataset {name}"):
        if name in {"Cora", "Citeseer", "PubMed"}:
            return _load_planetoid(name)
        if name in {"Chameleon", "Squirrel"}:
            return _load_wikipedia(name.lower())
        if name in {"ogbn-products", "ogbn-arxiv"}:
            g, _ = _load_ogb(name)
            return g
    raise RuntimeError(f"Unsupported dataset {name}")


# ---------------------------------------------------------------------------
#  Fast curvature estimator using GraphRicciCurvature (Sinkhorn mode)
# ---------------------------------------------------------------------------

def compute_sinkhorn_curvature(edge_index: torch.Tensor,
                               proj_dim: int = 32,
                               eps: float = 0.1,
                               device: str | torch.device = "cpu") -> torch.Tensor:
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError as e:
        raise RuntimeError("GraphRicciCurvature package missing – install via 'pip install GraphRicciCurvature'")

    src, dst = edge_index.cpu().numpy()
    g = nx.Graph()
    g.add_edges_from(zip(src, dst))
    with timer("Sinkhorn curvature"):
        orc = OllivierRicci(g, alpha=0, method="Sinkhorn", sinkhorn_epsilon=eps, dimension=proj_dim)
        orc.compute_ricci_curvature()
    kappa = []
    for (_, _, data) in g.edges(data=True):
        kappa.append(data.get("ricciCurvature", 0.0))
    return torch.tensor(kappa, dtype=torch.float32, device=device)
