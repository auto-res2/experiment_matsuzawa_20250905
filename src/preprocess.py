import inspect
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

def _build_ollivier_ricci(g: nx.Graph, proj_dim: int, eps: float):
    """Create an OllivierRicci instance while being robust to API changes.

    GraphRicciCurvature changed the keyword for the Sinkhorn epsilon several
    times (``epsilon`` vs ``sinkhorn_epsilon``).  We inspect the signature at
    runtime to select the correct one so that the code works with a wide range
    of library versions.
    """
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci  # noqa: WPS433
    except ImportError as err:
        raise RuntimeError(
            "GraphRicciCurvature package missing – install via 'pip install GraphRicciCurvature'"
        ) from err

    sig = inspect.signature(OllivierRicci.__init__)
    kwargs = {
        "alpha": 0,
        "method": "Sinkhorn",
    }
    if "sinkhorn_epsilon" in sig.parameters:
        kwargs["sinkhorn_epsilon"] = eps
    elif "epsilon" in sig.parameters:
        kwargs["epsilon"] = eps
    # Dimension / projection parameter also changed across versions
    if "dimension" in sig.parameters:
        kwargs["dimension"] = proj_dim
    elif "dim" in sig.parameters:
        kwargs["dim"] = proj_dim

    return OllivierRicci(g, **kwargs)


def compute_sinkhorn_curvature(
    edge_index: torch.Tensor,
    proj_dim: int = 32,
    eps: float = 0.1,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute Sinkhorn‐based Ollivier Ricci curvature for **each directed edge**.

    GraphRicciCurvature returns curvature for *undirected* edges (each unordered
    node pair once).  PyG, however, stores every undirected edge twice—once per
    direction.  To keep shapes consistent with ``edge_index`` we replicate the
    curvature value for both directions.
    """
    # ------------------------------------------------------------------
    #  Build NetworkX graph and run ORC
    # ------------------------------------------------------------------
    src, dst = edge_index.cpu().numpy()
    g = nx.Graph()
    g.add_edges_from(zip(src, dst))

    with timer("Sinkhorn curvature"):
        orc = _build_ollivier_ricci(g, proj_dim=proj_dim, eps=eps)
        orc.compute_ricci_curvature()

    # ------------------------------------------------------------------
    #  Create a mapping (u,v)->kappa for undirected edges
    # ------------------------------------------------------------------
    undirected_curv = {}
    for u, v, data in g.edges(data=True):
        key = tuple(sorted((u, v)))
        undirected_curv[key] = float(data.get("ricciCurvature", 0.0))

    # ------------------------------------------------------------------
    #  Replicate curvature for *each* directed edge in the original order
    # ------------------------------------------------------------------
    kappa_directed = [undirected_curv[tuple(sorted((int(u), int(v))))] for u, v in zip(src, dst)]
    return torch.tensor(kappa_directed, dtype=torch.float32, device=device)
