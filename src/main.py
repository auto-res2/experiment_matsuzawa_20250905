"""Orchestrates the whole experiment pipeline.
Launch via:  python -m src.main   (cwd should be project root)
"""
import json
import os
from pathlib import Path
from typing import Dict

import torch
import yaml

import preprocess as pp
from evaluate import save_line
from train import build_model, run_training

# ---------------------------------------------------------------------------
#  Directory structure (auto-created on first run)
# ---------------------------------------------------------------------------

RESEARCH_DIR = Path(".research") / "iteration2"
IMG_DIR = RESEARCH_DIR / "images"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  Load YAML config – if missing write a default one identical to the paper
# ---------------------------------------------------------------------------

CFG_PATH = Path("config") / "config.yaml"
CFG_PATH.parent.mkdir(exist_ok=True)

DEFAULT_CFG = {
    "tag": "depth_stress_cora",
    "dataset": {"name": "Cora"},
    "variants": [
        {"backbone": "GCN", "layers": l, "variant": "curvada"} for l in [2, 8, 32, 64, 120]
    ],
    "seeds": [0, 1, 2],
    "optim": {"lr": 5e-3, "weight_decay": 5e-4},
    "scheduler": {"max_epochs": 200},
    "curvada": {"alpha_init": 1.0, "beta_init": 0.0, "gamma_init": 1.0},
}
if not CFG_PATH.exists():
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump(DEFAULT_CFG, f)

with open(CFG_PATH) as f:
    cfg: Dict = yaml.safe_load(f)


# ---------------------------------------------------------------------------
#  Main logic
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------
    #  Load data + Sinkhorn curvature (edge & node-wise averages)
    # -------------------------------------------------------------------
    data = pp.get_dataset(cfg["dataset"]["name"])
    in_dim = data.x.size(-1)
    out_dim = int(data.y.max().item() + 1)

    kappa_edge = pp.compute_sinkhorn_curvature(data.edge_index, device=device)
    kappa_node = torch.zeros(data.num_nodes, device=device)
    src, dst = data.edge_index
    kappa_node.index_add_(0, src.to(device), kappa_edge)
    kappa_node.index_add_(0, dst.to(device), kappa_edge)
    deg = torch.bincount(src, minlength=data.num_nodes).to(device).clamp(min=1)
    kappa_node = kappa_node / deg

    # -------------------------------------------------------------------
    #  Results container – stored as JSON
    # -------------------------------------------------------------------
    results_json = RESEARCH_DIR / f"{cfg['tag']}.json"
    all_results = {"description": cfg}

    for variant in cfg["variants"]:
        name = f"{variant['variant']}_{variant['layers']}"
        print(f"\n===== Variant {name} =====")
        test_scores = []
        for seed in cfg["seeds"]:
            pp.set_seed(seed)
            model = build_model(
                backbone=variant["backbone"],
                in_channels=in_dim,
                out_channels=out_dim,
                num_layers=variant["layers"],
                variant=variant["variant"],
                curvada_hypers=cfg["curvada"],
            )
            optim = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, cfg["scheduler"]["max_epochs"])
            best_test = run_training(
                model=model,
                data=data,
                optimizer=optim,
                scheduler=sched,
                device=device,
                epochs=cfg["scheduler"]["max_epochs"],
                val_mask=data.val_mask,
                test_mask=data.test_mask,
                kappa_node=kappa_node,
            )
            test_scores.append(best_test)
        all_results[name] = {
            "mean": float(torch.tensor(test_scores).mean()),
            "std": float(torch.tensor(test_scores).std()),
            "all": test_scores,
        }
    with open(results_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print("==== JSON RESULTS ====")
    print(json.dumps(all_results, indent=2))

    # -------------------------------------------------------------------
    #  Plot accuracy vs depth
    # -------------------------------------------------------------------
    depths = [v["layers"] for v in cfg["variants"]]
    accs = [all_results[f"{v['variant']}_{v['layers']}"]["mean"] for v in cfg["variants"]]
    fig_path = IMG_DIR / f"accuracy_depth_{cfg['dataset']['name']}.pdf"
    save_line(depths, accs, "Depth", "Accuracy", f"Depth-Stress – {cfg['dataset']['name']}", fig_path.as_posix())
    print(f"Generated figure → {fig_path}")


if __name__ == "__main__":
    main()
