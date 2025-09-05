"""src/main.py
--------------------------------------------------------------------
Top-level orchestration script.  Only Experiment-1 is executed to keep
runtime reasonable within the evaluation constraints.

Run with
    python -m src.main
"""
from __future__ import annotations

# ----------------- standard lib ----------------------------------
import json
import os
from pathlib import Path
from typing import Dict, Any, List

# ----------------- third-party -----------------------------------
import yaml
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Sequential  # Sequential lives here

# -----------------------------------------------------------------
#  Robust import of reference models (GCN/GCNII)                    
# -----------------------------------------------------------------
try:
    from torch_geometric.nn.models import GCN, GCN2  # noqa: WPS433 – third-party optional import
except (ImportError, AttributeError):
    # The high-level model wrappers were removed in recent PyG versions.
    # We fall back to lightweight re-implementations to keep the code working.
    try:
        from torch_geometric.nn.models import GCN  # type: ignore
    except (ImportError, AttributeError):
        GCN = None  # type: ignore

    from torch_geometric.nn.conv import GCN2Conv

    class GCN2(torch.nn.Module):  # type: ignore
        """Minimal re-implementation of GCNII (a.k.a. GCN2 in PyG)."""

        def __init__(
            self,
            num_layers: int,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float = 0.5,
            alpha: float = 0.1,
            lamda: float = 0.5,
        ) -> None:
            super().__init__()
            self.dropout = float(dropout)
            self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
            self.convs = torch.nn.ModuleList(
                [GCN2Conv(hidden_channels, alpha, lamda, num_layers) for _ in range(num_layers)]
            )
            self.lin_out = torch.nn.Linear(hidden_channels, out_channels)
            self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        def forward(self, x, edge_index):  # noqa: D401 – standard forward signature
            x0 = x  # initial representation for residual fusion
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin_in(x).relu()
            for conv in self.convs:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = conv(x, x0, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin_out(x)
            return self.log_softmax(x)

# ----------------- project modules -------------------------------
from .preprocess import (
    load_planetoid,
    load_chameleon,
    load_ogbn_arxiv,
    set_seed,
)
from .train import CurvoNet, DropEdgeWrapper, train_model
from .evaluate import save_line_plot

# -----------------------------------------------------------------
#  Configuration                                                   
# -----------------------------------------------------------------
CFG_PATH = Path("config/config.yaml")
if not CFG_PATH.exists():
    raise FileNotFoundError("Configuration file missing – expected config/config.yaml")
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CONFIG: Dict[str, Any] = yaml.safe_load(f)

SEED = int(os.environ.get("GLOBAL_SEED", 0))
set_seed(SEED)


# -----------------------------------------------------------------
#  Experiment-1                                                    
# -----------------------------------------------------------------

def _build_model(model_name: str, data, depth: int) -> torch.nn.Module:
    """Return an instantiated model given its string identifier."""
    hidden = CONFIG["global"]["hidden_dim"]
    out_dim = int(data.y.max().item()) + 1
    in_dim = data.x.size(-1)

    if model_name == "GCN":
        if GCN is None:
            # fallback to a shallow manual implementation if import failed
            layers = []
            dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
            for i in range(depth):
                layers.append(GCNConv(dims[i], dims[i + 1]))
            return Sequential("x, edge_index", [(l, "x, edge_index -> x") for l in layers], torch.nn.LogSoftmax(dim=-1))
        return GCN(
            in_channels=in_dim,
            hidden_channels=hidden,
            num_layers=depth,
            out_channels=out_dim,
            dropout=0.5,
        )
    if model_name == "DropEdge":
        layers = []
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        for i in range(depth):
            conv = GCNConv(dims[i], dims[i + 1])
            layers.append(DropEdgeWrapper(conv, p=0.2))
        return Sequential("x, edge_index", [(l, "x, edge_index -> x") for l in layers], torch.nn.LogSoftmax(dim=-1))
    if model_name == "GCNII":
        return GCN2(
            num_layers=depth,
            in_channels=in_dim,
            hidden_channels=hidden,
            out_channels=out_dim,
        )
    if model_name == "CURVONet":
        return CurvoNet(in_dim=in_dim, hidden=hidden, out_dim=out_dim, num_layers=depth)

    raise ValueError(f"Model '{model_name}' not implemented")


def run_experiment_1(workdir: Path):
    print("\n===============================================================")
    print("Running Experiment 1 — Depth-Scaling Benchmark", flush=True)
    print("===============================================================")

    depths: List[int] = CONFIG["experiment_1"]["depths"]
    model_names: List[str] = CONFIG["experiment_1"]["models"]
    results: Dict[str, Dict[int, float]] = {m: {} for m in model_names}

    # loop over datasets -----------------------------------------
    for dname in CONFIG["experiment_1"]["datasets"]:
        print(f"\nDataset: {dname}")
        if dname == "Cora":
            data = load_planetoid("Cora", workdir / "data")
        elif dname == "Chameleon":
            data = load_chameleon(workdir / "data")
        elif dname == "ogbn-arxiv":
            data = load_ogbn_arxiv(workdir / "data")
        else:
            raise ValueError(f"Unknown dataset {dname}")

        for depth in depths:
            print(f"\n--- Depth L={depth} ---")
            for mname in model_names:
                set_seed(SEED)
                model = _build_model(mname, data, depth)

                run_cfg = {
                    "epochs": CONFIG["global"]["epochs"],
                    "early_stop_patience": CONFIG["global"]["early_stop_patience"],
                    "lr": 5e-3,
                    "weight_decay": 5e-4,
                    "mixed_precision": CONFIG["global"].get("mixed_precision", True),
                    "clip_grad": CONFIG["global"].get("clip_grad", 1.0),
                }

                result = train_model(
                    model,
                    data,
                    data.train_mask,
                    data.val_mask,
                    data.test_mask,
                    run_cfg,
                )
                print(
                    f"Model {mname:10s} | TestAcc {result.test_metric:.3f} "
                    f"| ValAcc {result.val_metric:.3f} | Time {result.train_time/60:.1f} min "
                    f"| FLOPs {result.flops_g:.2f} GF",
                    flush=True,
                )
                results[mname][depth] = result.test_metric

        # plotting ------------------------------------------------
        for mname, depth_dict in results.items():
            xs = sorted(depth_dict.keys())
            ys = [depth_dict[d] for d in xs]
            fname = f"accuracy_{dname}_{mname}.pdf"
            save_line_plot(xs, ys, f"{mname} on {dname}", "Test accuracy", fname)
            print(f"Saved figure {fname}")

    # raw json dump ----------------------------------------------
    out_path = workdir / "exp1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to {out_path}")


# -----------------------------------------------------------------
#  Entry-point                                                     
# -----------------------------------------------------------------

def main():  # noqa: D401 – simple CLI entry
    workdir = Path("./runs").absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    run_experiment_1(workdir)


if __name__ == "__main__":
    main()
