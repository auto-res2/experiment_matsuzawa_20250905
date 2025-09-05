"""src/train.py
--------------------------------------------------------------------
All model definitions and the generic training routine live here.
The module is intentionally independent from the rest of the project
except for the `config` dictionary that is passed in from src.main.
"""
from __future__ import annotations

# ----------------- standard lib ----------------------------------
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# ----------------- third-party -----------------------------------
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.utils import degree

try:
    from fvcore.nn import FlopCountAnalysis  # optional but nice to have
except ImportError:  # pragma: no cover
    FlopCountAnalysis = None  # type: ignore

# -----------------------------------------------------------------
#  Utilities                                                        
# -----------------------------------------------------------------

def simple_forman_curvature(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Cheap O(E) Forman-Ricci curvature approximation κ(u,v)=4−deg(u)−deg(v).
    Returns a vector of length |E| with curvature values (float32).
    """
    row, col = edge_index
    degs = degree(row, num_nodes=num_nodes).float()
    kappa = 4.0 - degs[row] - degs[col]
    return kappa


class ContraNorm(nn.Module):
    """Variance-preserving normalisation from ContraNorm (local variant)."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 – simple interface
        mu = x.mean(dim=0, keepdim=True)
        var = (x - mu).pow(2).mean(dim=0, keepdim=True)
        return (x - mu) / (var + self.eps).sqrt()


class CurvoLayer(nn.Module):
    """One layer of CURVONet implementing curvature gating + ContraNorm."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.norm = ContraNorm()
        self.gamma = nn.Parameter(torch.tensor(0.1))  # residual strength (pre-sigmoid)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  # noqa: D401
        kappa = simple_forman_curvature(edge_index, x.size(0)).unsqueeze(-1)  # [E,1]
        a_uv = self.gate_mlp(kappa).squeeze()                                # [E]
        out = self.gcn(x, edge_index, edge_weight=a_uv)
        out = self.norm(out)
        beta = torch.sigmoid(self.gamma)
        return (1 - beta) * F.relu(out) + beta * x


class CurvoNet(nn.Module):
    """Stack of CurvoLayers + plain GCN head."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([CurvoLayer(dims[i], dims[i + 1]) for i in range(num_layers - 1)])
        self.final_conv = GCNConv(dims[-2], dims[-1], add_self_loops=False, normalize=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  # noqa: D401
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.final_conv(x, edge_index)
        return self.log_softmax(x)


class DropEdgeWrapper(nn.Module):
    """Wrap a PyG GCNConv with edge-dropout at training time."""

    def __init__(self, conv: GCNConv, p: float = 0.2):
        super().__init__()
        self.conv = conv
        self.p = p

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:  # noqa: D401
        if self.training:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.p
            edge_index = edge_index[:, mask]
        return self.conv(x, edge_index)


@dataclass
class RunResult:
    test_metric: float
    val_metric: float
    train_time: float
    best_epoch: int
    flops_g: float


# -----------------------------------------------------------------
#  Training routine                                                
# -----------------------------------------------------------------

def train_model(
    model: nn.Module,
    data,
    train_mask: Tensor,
    val_mask: Tensor,
    test_mask: Tensor,
    config: Dict[str, Any],
) -> RunResult:
    """Generic full-batch training loop with early stopping."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = model.to(device), data.to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.get("mixed_precision", False) and device.type == "cuda")

    best_val, best_state, best_epoch = -1.0, None, 0
    start = time.time()

    for epoch in range(config["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[train_mask], data.y[train_mask])
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("clip_grad", 1.0))
        scaler.step(opt)
        scaler.update()

        # --- validation ----------------------------------------------------
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=-1)
        val_acc = (pred[val_mask] == data.y[val_mask]).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if epoch - best_epoch >= config.get("early_stop_patience", 100):
            break

    train_time = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    # final evaluation -------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=-1)
    test_acc = (pred[test_mask] == data.y[test_mask]).float().mean().item()

    # FLOPs (single forward pass) -------------------------------------------
    flops_g = 0.0
    if FlopCountAnalysis is not None:
        try:
            flops_g = FlopCountAnalysis(model, (data.x, data.edge_index)).total() / 1e9
        except Exception:
            flops_g = 0.0  # gracefully degrade

    return RunResult(
        test_metric=test_acc,
        val_metric=best_val,
        train_time=train_time,
        best_epoch=best_epoch,
        flops_g=flops_g,
    )
