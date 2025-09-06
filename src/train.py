import os
import math
import json
import time
from contextlib import contextmanager
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.nn import GCNConv, GATConv, SGConv, PairNorm
from torch_geometric.loader import NeighborLoader

# ---------------------------------------------------------------------------
#  Small utility helpers (kept local to avoid an extra utils file)
# ---------------------------------------------------------------------------

@contextmanager
def timer(msg: str):
    start = time.time()
    yield
    print(f"{msg} – {time.time() - start:.2f}s", flush=True)


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Node–wise classification accuracy."""
    if pred.ndim == 2:
        pred = pred.argmax(dim=-1)
    return float((pred == y).sum().item() / y.numel())

# ---------------------------------------------------------------------------
#  CurvAdaNorm building blocks & back-bones
# ---------------------------------------------------------------------------

class CurvAdaNormLayer(torch.nn.Module):
    """Curvature–adaptive gate + PairNorm-like normalisation.
    For brevity only the inference-time formulation is given.  The surrounding
    GNN layer is expected to multiply the incoming message by the pre-computed
    gate.  Here we only apply the variance-preserving node-wise rescale.
    """

    def __init__(self, feat_dim: int, alpha: float, beta: float, gamma: float):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        self.beta = torch.nn.Parameter(torch.tensor(beta))
        self.gamma = torch.nn.Parameter(torch.tensor(gamma))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor, kappa_node: torch.Tensor) -> torch.Tensor:
        # PairNorm style centering
        x = x - x.mean(dim=0, keepdim=True)
        # curvature weighted scaling
        scale = (1.0 + self.gamma * kappa_node[:, None]).clamp(min=self.eps)
        return x / scale


class _GCN(torch.nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = GCNConv(in_c, out_c, add_self_loops=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class _GAT(torch.nn.Module):
    def __init__(self, in_c: int, out_c: int, heads: int = 8):
        super().__init__()
        self.conv = GATConv(in_c, out_c // heads, heads=heads, concat=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class _SGC(torch.nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 1):
        super().__init__()
        self.conv = SGConv(in_c, out_c, K=k, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


# ---------------------------------------------------------------------------
#  Model factory – dynamically assembles a sequential model of arbitrary depth
# ---------------------------------------------------------------------------

class GraphModel(torch.nn.Module):
    """Light wrapper that routes arguments to the appropriate sub-modules."""

    def __init__(self, layers: List[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, kappa_node: torch.Tensor | None = None):
        for layer in self.layers:
            if isinstance(layer, CurvAdaNormLayer):
                if kappa_node is None:
                    raise RuntimeError("CurvAdaNormLayer requires 'kappa_node' input.  Make sure to pass it when calling the model.")
                x = layer(x, kappa_node)
            elif isinstance(layer, (PairNorm,)):
                x = layer(x)
            elif isinstance(layer, (_GCN, _GAT, _SGC)):
                x = layer(x, edge_index)
            else:  # linear layer
                x = layer(x)
        return x


def build_model(backbone: str,
                in_channels: int,
                out_channels: int,
                num_layers: int,
                variant: str,
                curvada_hypers: Dict[str, float]) -> torch.nn.Module:
    layers: List[torch.nn.Module] = []
    hidden = 128
    use_curvada = variant.startswith("curvada") or variant.startswith("ablation")

    BackBoneCls = {
        "GCN": _GCN,
        "GAT": _GAT,
        "SGC": _SGC,
    }[backbone]

    for l in range(num_layers):
        in_c = in_channels if l == 0 else hidden
        out_c = hidden
        if use_curvada:
            layers.append(
                CurvAdaNormLayer(
                    feat_dim=in_c,
                    alpha=curvada_hypers["alpha_init"],
                    beta=curvada_hypers["beta_init"],
                    gamma=curvada_hypers["gamma_init"],
                )
            )
        layers.append(BackBoneCls(in_c, out_c))
        if variant == "pairnorm":
            layers.append(PairNorm())
    layers.append(torch.nn.Linear(hidden, out_channels))
    return GraphModel(layers)


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def run_training(model: torch.nn.Module,
                 data,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 device: str,
                 epochs: int,
                 val_mask: torch.Tensor,
                 test_mask: torch.Tensor,
                 kappa_node: torch.Tensor | None = None,
                 log_dict: dict | None = None) -> float:
    """Core mini-batch-less full-graph training routine."""
    model.to(device)
    data = data.to(device)
    if kappa_node is not None:
        kappa_node = kappa_node.to(device)
    x, y = data.x, data.y.squeeze()
    edge_index = data.edge_index

    scaler = GradScaler(enabled=(device == "cuda"))
    best_val, best_test = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        with autocast(enabled=(device == "cuda")):
            logits = model(x, edge_index, kappa_node)
            loss = F.cross_entropy(logits[val_mask], y[val_mask])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits = model(x, edge_index, kappa_node)
                val_acc = accuracy(logits[val_mask], y[val_mask])
                test_acc = accuracy(logits[test_mask], y[test_mask])
            if val_acc > best_val:
                best_val, best_test = val_acc, test_acc
            print(f"Epoch {epoch:4d} | loss {loss.item():.4f} | val {val_acc:.4f} | test {test_acc:.4f}")

    if log_dict is not None:
        log_dict["best_val"] = best_val
        log_dict["best_test"] = best_test
    return best_test
