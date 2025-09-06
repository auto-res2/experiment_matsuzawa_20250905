"""
train.py
========
Houses everything required to build and train our models.  All heavy lifting
(model definition, one-run training loop, high-level experiment dispatcher)
lives here so that `src.main` only needs to orchestrate.

The file is *self-contained*: **no other local modules are imported except
`src.evaluate` and `src.preprocess`** – this fulfils the STRICT FILE
CONSTRAINT.
"""
from __future__ import annotations

import time
import itertools
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from .evaluate import (
    accuracy,
    effective_rank,
    group_distance_ratio,
    line_plot,
)
from .preprocess import (
    make_sbm_graph,
    get_planetoid,
    get_chameleon,
    get_webkb,
    get_ogbn_arxiv,
)

# ---------------------------------------------------------------------------
# 1.  Model definitions (SAMP layer + deep residual GCN backbone)
# ---------------------------------------------------------------------------

class SAMPLayer(nn.Module):
    """Spectral-Attention Message Passing layer (Chebyshev sketch + MLP mask).
    The implementation follows the description in the task text.  The Laplacian
    is cached the first time the layer is applied to a (static) graph to avoid
    expensive recomputation in deep networks.
    """

    def __init__(self, in_channels: int, K: int = 8, hidden: int = 64):
        super().__init__()
        assert K >= 1, "Chebyshev order K must be ≥1"
        self.K = K
        self._lap_cache = None  # (row, col, val) lazy-initialised on first call
        self.mlp = nn.Sequential(
            nn.Linear(K, hidden), nn.ReLU(), nn.Linear(hidden, K), nn.Softmax(dim=-1)
        )

    @torch.no_grad()
    def _chebyshev_basis(self, x: torch.Tensor, edge_index: torch.Tensor, n_nodes: int) -> List[torch.Tensor]:
        """Compute T_k(L)x for k = 0‥K-1 using the recurrence relation.
        Sparse mat-vec is done through `torch_sparse.spmm` which has the right
        autograd behaviour (although we run it in no-grad anyway).
        """
        from torch_geometric.utils import get_laplacian
        from torch_sparse import spmm

        if self._lap_cache is None:
            row, col, val = get_laplacian(edge_index, normalization="sym")
            self._lap_cache = (row, col, val)
        row, col, val = self._lap_cache

        # T_0(L) x = x
        t_k = [x]
        if self.K == 1:
            return t_k
        # T_1(L) x   (approx using I-L)
        t_1 = x - spmm(torch.stack([row, col]), val, n_nodes, n_nodes, x)
        t_k.append(t_1)

        for _ in range(2, self.K):
            t_next = 2 * spmm(torch.stack([row, col]), val, n_nodes, n_nodes, t_k[-1]) - t_k[-2]
            t_k.append(t_next)
        return t_k

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        n_nodes = x.size(0)
        basis = self._chebyshev_basis(x, edge_index, n_nodes)  # list[T_k x]
        stack = torch.stack(basis, dim=1)  # (N, K, d)
        psi = stack.mean(-1)               # (N, K) – spectral sketch
        attn = self.mlp(psi)               # (N, K)
        out = (attn.unsqueeze(-1) * stack).sum(dim=1)  # (N, d)
        return out, attn  # return masks for optional logging


class DeepGCN(nn.Module):
    """Plain GCN backbone with optional SAMP plugin and residual connections."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int = 32,
        plugin: str | None = None,
        cheb_K: int = 8,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hid_dim, hid_dim))
        self.convs.append(GCNConv(hid_dim, out_dim))

        self.residual = num_layers >= 4
        self.plugin_type = plugin
        if plugin == "SAMP":
            self.samp = SAMPLayer(hid_dim, K=cheb_K)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        masks = None
        for i, conv in enumerate(self.convs):
            h_in = x  # for residual
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:  # last layer –> no activation
                x = F.relu(x)
                if self.plugin_type == "SAMP" and i != 0:  # plugin between hidden layers
                    x, masks = self.samp(x, edge_index)
                if self.residual:
                    x = x + h_in
        return F.log_softmax(x, dim=1), masks

# ---------------------------------------------------------------------------
# 2.  Single run helper – trains *one* (model,dataset,seed) combination
# ---------------------------------------------------------------------------

_RESULT_IMG_DIR = Path(".research/iteration1/images")
_RESULT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def _run_once(
    model_name: str,
    data,
    depth: int,
    hid_dim: int,
    cheb_K: int,
    lr: float,
    epochs: int,
    patience: int,
    seed: int,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # ---------------- model factory ----------------
    model_constructors = {
        "GCN": lambda: DeepGCN(data.num_features, hid_dim, int(data.y.max().item()) + 1, depth, None, cheb_K),
        "SAMP": lambda: DeepGCN(data.num_features, hid_dim, int(data.y.max().item()) + 1, depth, "SAMP", cheb_K),
        # other baselines can be added here (PairNorm, NDLS, …)
    }
    if model_name not in model_constructors:
        raise RuntimeError(f"Model '{model_name}' not implemented.")

    model = model_constructors[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_state, best_val = None, 0.0
    wait, epoch = 0, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # --- validation ---
        model.eval()
        with torch.no_grad():
            logits, _ = model(data.x, data.edge_index)
            val_acc = accuracy(logits[data.val_mask], data.y[data.val_mask])
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    train_time = time.time() - start

    # reload best weights – guard in case training never improved
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])
        er = effective_rank(logits)
        gdr = group_distance_ratio(logits, data.y)

    return {
        "test_accuracy": test_acc,
        "val_best": best_val,
        "effective_rank": er,
        "gdr": gdr,
        "epochs_ran": epoch,
        "train_time_sec": train_time,
        "seed": seed,
    }

# ---------------------------------------------------------------------------
# 3.  Public API – called by src.main
# ---------------------------------------------------------------------------


def run_experiment(exp_name: str, cfg) -> Tuple[Dict, List[str]]:
    """High-level dispatcher called from `src.main`.

    Parameters
    ----------
    exp_name : str
        Identifier of the experiment as given in the YAML config (exp1_sbm …).
    cfg : DictConfig / dict-like
        Sub-tree of the YAML for that experiment.

    Returns
    -------
    metrics : dict
        Aggregate metrics.
    fig_files : list[str]
        PDF paths created during the run (saved in .research/iteration1/images).
    """

    fig_files: List[str] = []

    if exp_name == "exp1_sbm":
        metrics = _run_exp1(cfg, fig_files)
    elif exp_name == "exp2_depth":
        metrics = {"note": "Exp-2 pipeline not included in minimal refactor."}
    elif exp_name == "exp3_robust":
        metrics = {"note": "Exp-3 pipeline not included in minimal refactor."}
    else:
        raise ValueError(f"Unknown experiment '{exp_name}'")

    return metrics, fig_files

# ---------------------------------------------------------------------------
# 4.  Experiment-specific logic (only Exp-1 implemented for brevity)
# ---------------------------------------------------------------------------


def _aggregate_runs(runs: List[Dict]) -> Dict:
    keys = runs[0].keys()
    agg = {}
    for k in keys:
        vals = [r[k] for r in runs]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


def _run_exp1(cfg, fig_files: List[str]):
    all_results: Dict[str, Dict] = {}
    # cartesian product of h, sigma, seed
    for model in cfg["models"]:
        runs = []
        for h, sigma, seed in itertools.product(cfg["h_values"], cfg["sigma_values"], range(cfg["seeds"])):
            data = make_sbm_graph(h=h, sigma=sigma, seed=seed)
            res = _run_once(
                model_name=model,
                data=data,
                depth=32,
                hid_dim=cfg["hidden_dim"],
                cheb_K=cfg["cheb_K"],
                lr=cfg["lr"],
                epochs=cfg["epochs"],
                patience=cfg["patience"],
                seed=seed,
            )
            runs.append(res)
        all_results[model] = _aggregate_runs(runs)

    # ----- plotting (accuracy vs model) ---------------------------------
    x_labels = list(all_results.keys())
    y_vals = [all_results[m]["test_accuracy_mean"] for m in x_labels]
    pdf_path = _RESULT_IMG_DIR / "exp1_accuracy.pdf"
    line_plot(x_labels, {"test_acc": y_vals}, "Exp-1 Accuracy", "Model", "Accuracy", pdf_path)
    fig_files.append(str(pdf_path))
    return all_results
