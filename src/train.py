from __future__ import annotations

import os
import random
import time
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Torch-Geometric -----------------------------------------------------------
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

# Optional – FLOP counter (fails gracefully if absent) ----------------------
try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
except ImportError:  # pragma: no cover – fvcore is optional
    FlopCountAnalysis = None

# Local imports -------------------------------------------------------------
from .evaluate import average_pairwise_distance, effective_rank  # no circular deps

__all__ = [
    "ContraNorm",
    "CurvoLayer",
    "CurvoNet",
    "PlainGCN",
    "DropEdgeGCN",
    "GCNII",
    "build_model",
    "train",
]

# ---------------------------------------------------------------------------
# Normalisation & curvature helpers
# ---------------------------------------------------------------------------
class ContraNorm(nn.Module):
    """Variance-preserving normalisation (local variant as in the paper)."""

    def __init__(self, eps: float = 1e-5) -> None:  # noqa: D401 – trivial doc
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 – simple forward
        mu = x.mean(dim=0, keepdim=True)
        var = (x - mu).pow(2).mean(dim=0, keepdim=True)
        return (x - mu) / (var + self.eps).sqrt()


# ---------------------------------------------------------------------------
# Curvature helper
# ---------------------------------------------------------------------------

def _forman_curvature(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Cheap O(E) Forman curvature approximation κ = 4 − deg(u) − deg(v)."""
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)
    return 4.0 - deg[row] - deg[col]


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class CurvoLayer(nn.Module):
    """Single CURVONet layer: curvature-gated GCN + ContraNorm + residual."""

    def __init__(self, in_dim: int, out_dim: int) -> None:  # noqa: D401
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        self.gate = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.norm = ContraNorm()
        self.gamma = nn.Parameter(torch.tensor(0.1))  # learnable residual coefficient

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  # noqa: D401
        kappa = _forman_curvature(edge_index, x.size(0)).unsqueeze(-1)  # [E,1]
        att = self.gate(kappa).squeeze()  # [E]
        h = self.conv(x, edge_index, edge_weight=att)
        h = self.norm(h)
        beta = torch.sigmoid(self.gamma)
        return (1 - beta) * F.relu(h) + beta * x


class CurvoNet(nn.Module):
    """Full CURVONet backbone."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [CurvoLayer(in_dim if i == 0 else hidden, hidden) for i in range(num_layers)]
        )
        self.head = GCNConv(hidden, out_dim, add_self_loops=False, normalize=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  # noqa: D401
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.head(x, edge_index)
        return self.logsoftmax(x)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class DropEdgeGCN(nn.Module):
    """GCN with edge dropout (DropEdge baseline)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int, p: float = 0.2):
        super().__init__()
        self.p = p
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  # noqa: D401
        if self.training:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.p
            edge_index = edge_index[:, mask]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return self.logsoftmax(x)


class PlainGCN(nn.Module):
    """Vanilla N-layer GCN."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  # noqa: D401
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
        return self.logsoftmax(x)


class GCNII(nn.Module):
    """Minimal, AMP-friendly re-implementation of GCNII (a.k.a. GCN2)."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        num_layers: int,
        alpha: float = 0.1,
        lamda: float = 0.5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GCN2Conv  # local import avoids hard version pin

        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [GCN2Conv(hidden, alpha, lamda, num_layers) for _ in range(num_layers)]
        )
        self.lin_out = nn.Linear(hidden, out_dim)
        self.dropout = float(dropout)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  # noqa: D401
        x0 = x.detach()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_in(x))
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)
        return self.logsoftmax(x)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Module:
    """Return the requested model instance given its identifier string."""
    name = name.upper()
    if name == "GCN":
        return PlainGCN(in_dim, hidden, out_dim, depth)
    if name == "DROPEdge".upper():
        return DropEdgeGCN(in_dim, hidden, out_dim, depth)
    if name == "GCNII":
        return GCNII(in_dim, hidden, out_dim, depth)
    if name == "CURVONET":
        return CurvoNet(in_dim, hidden, out_dim, depth)
    raise ValueError(f"Model '{name}' not recognised")


# ---------------------------------------------------------------------------
# Training loop (full-batch)
# ---------------------------------------------------------------------------

def train(model: nn.Module, data, cfg: SimpleNamespace) -> Dict:  # noqa: ANN001 – PyG data
    """Full-batch training with early stopping and automatic mixed precision."""

    device = torch.device(cfg.device)
    model = model.to(device)
    data = data.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

    best_val = -1.0
    best_state: Dict[str, Tensor] | None = None
    best_epoch = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        # Validation -------------------------------------------------------
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=-1)
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if epoch - best_epoch >= cfg.early_stop_patience:
            break

    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation ------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=-1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    # Secondary metrics -----------------------------------------------------
    h_last = logits.detach()
    apd = average_pairwise_distance(h_last[data.test_mask].float())
    reff = effective_rank(h_last[data.test_mask].float())

    # FLOP count (best-effort) ---------------------------------------------
    flops = 0.0
    if FlopCountAnalysis is not None:
        try:
            flops = FlopCountAnalysis(model, (data.x, data.edge_index)).total() / 1e9
        except Exception:  # pragma: no cover – unsupported op
            flops = 0.0

    return {
        "test_acc": test_acc,
        "val_acc": best_val,
        "epochs": epoch + 1,
        "time_s": train_time,
        "apd": apd,
        "reff": reff,
        "flops_g": flops,
    }
