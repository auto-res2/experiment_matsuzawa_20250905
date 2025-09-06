"""
train.py – model architectures and training loop for CurvAdaNorm experiments
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import PairNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, dropout_edge
import torch.nn.functional as F  # NEW

from .evaluate import accuracy  # evaluation metrics used inside the training loop

# ----------------------------------------------------------------------------
#  Model components
# ----------------------------------------------------------------------------

class CurvGCNConv(MessagePassing):
    """GCN-style message passing layer with curvature-aware edge gates."""

    def __init__(self, in_c: int, out_c: int, bias: bool = True):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_c, out_c, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_c))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        kappa_edge: torch.Tensor,
        dropedge_p: float = 0.0,
    ) -> torch.Tensor:
        # ------------------------------------------------------------------
        # Edge dropout – ensure that curvature attributes are dropped *together*
        # with their corresponding edges to avoid length mismatches.
        # ------------------------------------------------------------------
        if self.training and dropedge_p > 0.0:
            # dropout_edge now returns (edge_index, edge_attr, mask) in recent
            # PyG releases.  We therefore unpack accordingly and fall back to the
            # older 2-tuple behaviour if only two values are returned.
            out = dropout_edge(
                edge_index,
                p=dropedge_p,
                force_undirected=True,
                training=True,
            )
            if len(out) == 3:
                edge_index, _, edge_mask = out
            else:
                edge_index, edge_mask = out  # type: ignore[misc]
            kappa_edge = kappa_edge[edge_mask]

        # ------------------------------------------------------------------
        # Add self-loops; curvature for those edges is defined as 0.
        # PyG's helper keeps edge_attr in sync so no manual slicing needed.
        # ------------------------------------------------------------------
        edge_index, kappa_edge = add_self_loops(
            edge_index,
            edge_attr=kappa_edge,
            fill_value=0.0,
            num_nodes=x.size(0),
        )

        gate = torch.sigmoid(self.alpha * kappa_edge.float() + self.beta)

        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        norm = (deg.pow(-0.5))[row] * (deg.pow(-0.5))[col]

        return self.propagate(edge_index, x=self.lin(x), norm=norm, gate=gate)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return norm.unsqueeze(-1) * gate.unsqueeze(-1) * x_j

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class CurvAdaNorm(nn.Module):
    """PairNorm-style centring + curvature-weighted scaling."""

    def __init__(self, gamma_init: float = 0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.eps = 1.0e-6

    def forward(self, x: torch.Tensor, kappa_node: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        scale = (1.0 + self.gamma.clamp(-3, 3) * kappa_node.unsqueeze(-1)).clamp(min=self.eps)
        return x / scale


class DGNLayer(nn.Module):
    """Deep Graph Normalisation (simplified, default groups = 2)."""

    def __init__(self, feat_dim: int, num_groups: int = 2):
        super().__init__()
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(feat_dim))
        self.eps = 1.0e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.shape
        # Reshape into groups * batch * feat
        x_g = x.reshape(self.num_groups, -1, c)
        mean = x_g.mean(dim=1, keepdim=True)
        var = x_g.var(dim=1, unbiased=False, keepdim=True)
        x = (x_g - mean) / torch.sqrt(var + self.eps)
        return x.reshape(n, c) * self.weight


class PSNRGate(nn.Module):
    """Learnable residual gate used by the PSNR baseline."""

    def __init__(self, feat_dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))
        self.lin = nn.Linear(feat_dim, feat_dim)

    def forward(self, x_new: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + torch.sigmoid(self.gate) * self.lin(x_new)


class GNNStack(nn.Module):
    """Back-bone + normalisation variants assembled automatically."""

    def __init__(self, in_dim: int, out_dim: int, depth: int, variant: str, dropedge_p: float):
        super().__init__()
        hidden = 128
        layers: List[nn.Module] = []
        for l in range(depth):
            if variant == "curvada":
                layers.append(CurvAdaNorm())
                layers.append(CurvGCNConv(hidden if l else in_dim, hidden))
            elif variant == "pairnorm":
                layers.append(PairNorm())
                layers.append(CurvGCNConv(hidden if l else in_dim, hidden))
            elif variant == "dgn":
                layers.append(DGNLayer(hidden if l else in_dim))
                layers.append(CurvGCNConv(hidden if l else in_dim, hidden))
            elif variant == "psnr":
                layers.append(CurvGCNConv(hidden if l else in_dim, hidden))
            else:  # vanilla GCN baseline
                layers.append(CurvGCNConv(hidden if l else in_dim, hidden))
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden, out_dim)
        self.variant = variant
        self.dropedge_p = dropedge_p
        if variant == "psnr":
            self.psnr_gates = nn.ModuleList([PSNRGate(hidden) for _ in range(depth)])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        kappa_edge: torch.Tensor,
        kappa_node: torch.Tensor,
    ) -> torch.Tensor:
        psnr_idx = 0
        for layer in self.layers:
            if isinstance(layer, CurvGCNConv):
                x_new = layer(x, edge_index, kappa_edge, dropedge_p=self.dropedge_p)

                if self.variant == "psnr":
                    # Ensure dimensionality match for residual connection.
                    residual = x if x.shape[-1] == x_new.shape[-1] else torch.zeros_like(x_new)
                    x = self.psnr_gates[psnr_idx](x_new, residual)
                    psnr_idx += 1
                else:
                    # Vanilla / other variants – add residual only if dims match
                    if x.shape[-1] == x_new.shape[-1]:
                        x = x_new + x
                    else:
                        x = x_new

                # --- NEW: Non-linearity for numerical stability ---
                x = F.relu(x)

            elif isinstance(layer, CurvAdaNorm):
                x = layer(x, kappa_node)
            elif isinstance(layer, PairNorm):
                x = layer(x)
            elif isinstance(layer, DGNLayer):
                x = layer(x)
        return self.classifier(x)

# ----------------------------------------------------------------------------
#  Training utilities
# ----------------------------------------------------------------------------

class EarlyStopper:
    """Simple validation-based early stopping."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = None
        self.counter = 0

    def step(self, metric: float) -> bool:
        if self.best is None or metric > self.best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


@torch.no_grad()
def _eval(
    model: nn.Module, data, kappa_edge, kappa_node, device: str
) -> Tuple[float, float]:
    model.eval()
    out = model(data.x, data.edge_index, kappa_edge, kappa_node)
    val_acc = accuracy(out[data.val_mask], data.y[data.val_mask])
    test_acc = accuracy(out[data.test_mask], data.y[data.test_mask])
    return val_acc, test_acc


def train_single(
    data,
    kappa_edge,
    kappa_node,
    depth: int,
    variant: str,
    device: str,
    cfg: Dict,
) -> Tuple[float, float]:
    """Train on a single (pre-split) dataset split and return (test_acc, best_val_acc)."""

    num_classes = int(data.y.max().item()) + 1
    model = GNNStack(
        data.num_features, num_classes, depth, variant, dropedge_p=0.2 if depth > 32 else 0.0
    ).to(device)

    data = data.to(device, non_blocking=True)
    kappa_edge, kappa_node = kappa_edge.to(device), kappa_node.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=5e-4)
    sched = CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
    scaler = GradScaler(enabled=(device == "cuda"))
    stopper = EarlyStopper(patience=50)

    best_val, best_test = 0.0, 0.0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        opt.zero_grad()
        with autocast(enabled=(device == "cuda")):
            out = model(data.x, data.edge_index, kappa_edge, kappa_node)
            loss = torch.nn.functional.cross_entropy(
                out[data.train_mask], data.y[data.train_mask]
            )
        if torch.isnan(loss) or torch.isinf(loss):  # stronger check
            raise RuntimeError("NaN/Inf encountered during training – aborting.")
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        # evaluation every 5 epochs for speed
        if epoch % 5 == 0 or epoch == cfg["max_epochs"]:
            val_acc, test_acc = _eval(model, data, kappa_edge, kappa_node, device)
            if val_acc > best_val:
                best_val, best_test = val_acc, test_acc
            if stopper.step(val_acc):
                break

    return best_test, best_val
