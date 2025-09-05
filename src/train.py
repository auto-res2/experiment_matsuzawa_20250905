"""
train.py – model definitions and training utilities for AFN-GNN experimental suite
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .evaluate import (
    accuracy,
    group_distance_ratio,
    instance_information_gain,
)
from .preprocess import set_seed  # re-exported for convenience

# ──────────────────────────────────────────────────────────────────────────────
#  Helper: sparse-dense matrix multiplication (Chebyshev recursion)
# ──────────────────────────────────────────────────────────────────────────────

def torch_sparse_spmm(index, weight, N, X):
    """Sparse COO (index, weight)  ×  dense matrix X => torch.Tensor"""
    L = torch.sparse_coo_tensor(index, weight, (N, N))
    return torch.sparse.mm(L, X)


# ──────────────────────────────────────────────────────────────────────────────
#  Filter bank & mixers
# ──────────────────────────────────────────────────────────────────────────────

class ChebFilterBank:
    """Compute Chebyshev polynomial filters T_k(\tilde{L}) on-the-fly."""

    def __init__(self, data, K: int):
        self.K = K
        self.L_index = data.lap_edge_index
        self.L_weight = data.lap_edge_weight
        self.num_nodes = data.num_nodes

    def apply(self, X: torch.Tensor, K: int | None = None) -> List[torch.Tensor]:
        K = self.K if K is None else K
        outs = [X]
        if K == 0:
            return outs
        # T1 = L̃X
        T_1 = torch_sparse_spmm(self.L_index, self.L_weight, self.num_nodes, X)
        outs.append(T_1)
        for _ in range(2, K + 1):
            T_k = 2 * torch_sparse_spmm(self.L_index, self.L_weight, self.num_nodes, outs[-1]) - outs[-2]
            outs.append(T_k)
        return outs


class AdaptiveSpectralMixer(nn.Module):
    """Node-wise softmax gate over K+1 frequency bands."""

    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.K = K
        self.gate = nn.Sequential(
            nn.Linear(in_dim + 3, 2 * in_dim),
            nn.ReLU(),
            nn.Linear(2 * in_dim, K + 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        cheb_outputs: List[torch.Tensor],
        deg: torch.Tensor,
        curv: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aux = torch.stack(
            [deg, curv, torch.full_like(deg, layer_id, dtype=torch.float32)], dim=-1
        )
        logits = self.gate(torch.cat([h, aux], dim=-1))
        w = torch.softmax(logits, dim=-1)
        out = torch.zeros_like(h)
        for k in range(self.K + 1):
            out += w[:, k : k + 1] * cheb_outputs[k]
        return out, w


# ──────────────────────────────────────────────────────────────────────────────
#  Model definitions
# ──────────────────────────────────────────────────────────────────────────────

class AFNGNN(nn.Module):
    """Adaptive Frequency-band Node-wise GNN."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_classes: int,
        depth: int,
        K: int,
        dropout: float,
    ):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.K = K
        self.input_proj = nn.Linear(in_dim, hidden, bias=False)
        self.mixers = nn.ModuleList([AdaptiveSpectralMixer(hidden, K) for _ in range(depth)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden, num_classes)

    def forward(self, data, bank: ChebFilterBank):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        h = self.input_proj(x)
        deg = data.deg.to(h.device)
        curv = data.node_curv.to(h.device)
        all_weights = []
        for layer, (mixer, bn) in enumerate(zip(self.mixers, self.batch_norms)):
            cheb_outs = bank.apply(h, self.K)
            h_new, w = mixer(h, cheb_outs, deg, curv, layer + 1)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            beta_l = math.log(layer + 2) / (layer + 2)
            h = h_new + beta_l * h
            all_weights.append(w.detach())
        out = self.out_proj(h)
        return out, h, all_weights


class ChebGCN(nn.Module):
    """Baseline Chebyshev GCN."""

    def __init__(self, in_dim, hidden, num_classes, depth: int, K: int, dropout: float):
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.K = K
        self.input_proj = nn.Linear(in_dim, hidden, bias=False)
        self.theta = nn.ParameterList(
            [nn.Parameter(torch.empty(K + 1, hidden, hidden)) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(hidden, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.theta:
            nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, data, bank: ChebFilterBank):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        h = self.input_proj(x)
        for layer in range(self.depth):
            cheb_outs = bank.apply(h, self.K)
            tmp = 0.0
            for k, T_k_h in enumerate(cheb_outs):
                tmp += T_k_h @ self.theta[layer][k]
            h = F.relu(tmp)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.out_proj(h)
        return out, h, None


# ──────────────────────────────────────────────────────────────────────────────
#  GCNII wrapper (repository cloned on-the-fly)
# ──────────────────────────────────────────────────────────────────────────────

_GCNII_REPO_DIR = Path("external/GCNII")


def _ensure_gcnII(repo_url: str):
    if not _GCNII_REPO_DIR.exists():
        print("Cloning GCNII repository …")
        _GCNII_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"git clone {repo_url} {_GCNII_REPO_DIR}")
    sys.path.append(str(_GCNII_REPO_DIR))
    try:
        from model import GCNII as GCNIIModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import GCNII model") from exc
    return GCNIIModel


class GCNIIWrapper(nn.Module):
    """Thin wrapper exposing the same forward signature as other models."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_classes: int,
        depth: int,
        dropout: float,
        repo_url: str,
    ):
        super().__init__()
        GCNIIModel = _ensure_gcnII(repo_url)
        self.model = GCNIIModel(
            nfeat=in_dim,
            nlayers=depth,
            nhid=hidden,
            nclass=num_classes,
            dropout=dropout,
            lamda=0.5,
            alpha=0.1,
            variant=False,
        )

    def forward(self, data, bank=None):  # bank unused
        out = self.model(data.x, data.edge_index)
        return out, None, None


# ──────────────────────────────────────────────────────────────────────────────
#  Training loop (shared by all experiments)
# ──────────────────────────────────────────────────────────────────────────────

def train_one(
    model: nn.Module,
    data,
    bank: ChebFilterBank,
    cfg: dict,
) -> Tuple[float, float, float]:
    """Return (test_acc, GDR, IIG) at best validation epoch."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = model.to(device), data.to(device)
    bank_device = ChebFilterBank(data, bank.K)  # recreate on GPU / CPU

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=(cfg["optim"]["beta1"], cfg["optim"]["beta2"]),
    )
    scaler = (
        torch.cuda.amp.GradScaler(init_scale=2 ** 16)
        if cfg.get("precision", "fp32") == "fp16" and device.type == "cuda"
        else None
    )

    best_val, best_metrics, patience = -1.0, None, 0
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    for _ in range(cfg["num_epochs"]):
        model.train()
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            logits, emb, _ = model(data, bank_device)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # validation ----------------------------------------------------------
        model.eval()
        with torch.no_grad():
            logits, emb, _ = model(data, bank_device)
            val_acc = accuracy(logits[val_mask], data.y[val_mask])
            if val_acc > best_val:
                best_val = val_acc
                test_acc = accuracy(logits[test_mask], data.y[test_mask])
                gdr = group_distance_ratio(emb, data.y)
                iig = instance_information_gain(emb, data.y, int(data.y.max()) + 1)
                best_metrics = (test_acc, gdr, iig)
                patience = 0
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    break

    if best_metrics is None:
        raise RuntimeError("Training did not improve – check configuration.")
    return best_metrics
