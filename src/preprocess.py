from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

# Torch-Geometric -----------------------------------------------------------
import torch_geometric as tg
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import degree, to_undirected

try:
    from ogb.nodeproppred import PygNodePropPredDataset  # type: ignore
except ImportError:  # pragma: no cover – OGB is optional
    PygNodePropPredDataset = None

# ---------------------------------------------------------------------------
# Directories (created once at import time)
# ---------------------------------------------------------------------------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RUN_DIR = ROOT / "runs"
for _p in (DATA_DIR, FIG_DIR, RUN_DIR):
    _p.mkdir(parents=True, exist_ok=True)

__all__ = [
    "DATA_DIR",
    "FIG_DIR",
    "RUN_DIR",
    "set_seed",
    "load_dataset",
]

# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy (if present) and PyTorch RNGs for full reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Dataset loader & basic preprocessing
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> Tuple[tg.data.Data, int]:
    """Return PyG Data object and number of classes for the requested benchmark."""

    path = DATA_DIR / name.lower()
    name_low = name.lower()

    if name_low in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(root=str(path), name=name_low.capitalize())
        data = ds[0]
    elif name_low in {"chameleon", "squirrel", "texas"}:
        ds = WikipediaNetwork(root=str(path), name=name_low, geom_gcn_preprocess=False)
        data = ds[0]
    elif name_low == "ogbn-arxiv":
        if PygNodePropPredDataset is None:
            raise RuntimeError("OGB package missing – install ogb>=1.3.6 for OGB datasets")
        ds = PygNodePropPredDataset(name="ogbn-arxiv", root=str(path))
        data = ds[0]
        split = ds.get_idx_split()
        masks = {
            k: torch.zeros(data.num_nodes, dtype=torch.bool, device=data.edge_index.device)
            for k in ("train", "val", "test")
        }
        for k in masks:
            masks[k][split[k if k != "val" else "valid"].to(torch.long)] = True
        data.train_mask, data.val_mask, data.test_mask = masks.values()
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: Planetoid, WikipediaNetwork, ogbn-arxiv."
        )

    # Standard preprocessing ----------------------------------------------
    data = tg.data.Data.from_dict(data.__dict__)  # copy to allow edits
    data.edge_index = to_undirected(data.edge_index)

    # Row-normalise features safely
    row_sum = data.x.sum(dim=1, keepdim=True).clamp_min_(1e-12)
    data.x = data.x / row_sum

    # Ensure boolean masks (WikipediaNetwork provides index tensors instead)
    if data.train_mask.dtype != torch.bool:
        tmp = torch.zeros(data.num_nodes, dtype=torch.bool)
        tmp[data.train_mask] = True
        data.train_mask = tmp
        tmp = torch.zeros_like(tmp)
        tmp[data.val_mask] = True
        data.val_mask = tmp
        tmp = torch.zeros_like(tmp)
        tmp[data.test_mask] = True
        data.test_mask = tmp

    num_classes = int(data.y.max().item() + 1)
    return data, num_classes
