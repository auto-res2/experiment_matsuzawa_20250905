from __future__ import annotations
"""
main.py – orchestration entry-point (python -m src.main)
"""

import json
from pathlib import Path

import torch
import yaml

from .preprocess import compute_curvature, load_dataset, set_seed
from .train import train_single
from .evaluate import lineplot

# -----------------------------------------------------------------------------
#  Global configuration & paths (iteration *7* as per submission guidelines)
# -----------------------------------------------------------------------------

CONFIG_PATH = Path("config/config.yaml")
CFG = yaml.safe_load(CONFIG_PATH.read_text())

RESULTS_DIR = Path(".research/iteration7")
IMAGES_DIR = RESULTS_DIR / "images"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_exp1():
    cfg = CFG["exp1"]
    results = {"description": cfg["description"]}

    for ds_name in cfg["datasets"]:
        data = load_dataset(ds_name)
        print(f"Loaded {ds_name} – nodes: {data.num_nodes}, edges: {data.edge_index.size(1)}")
        kappa_e, kappa_n = compute_curvature(data, device=DEVICE)

        for depth in cfg["depths"]:
            for variant in cfg["variants"]:
                key = f"{ds_name}_{variant}_{depth}"
                scores = []
                for seed in cfg["seeds"]:
                    set_seed(seed)
                    test_acc, _ = train_single(
                        data, kappa_e, kappa_n, depth, variant, DEVICE, cfg
                    )
                    scores.append(test_acc)
                mean = float(torch.tensor(scores).mean().item())
                std = float(torch.tensor(scores).std().item())
                results[key] = {"mean": mean, "std": std, "all": scores}
                print(f"{key}: {mean:.4f} ± {std:.4f}")

        # Plot CurvAdaNorm accuracy vs depth.
        depths = cfg["depths"]
        ys = [results[f"{ds_name}_curvada_{d}"]["mean"] for d in depths]
        plot_path = IMAGES_DIR / f"accuracy_{ds_name}.pdf"
        lineplot(depths, ys, "Depth", "Accuracy", f"CurvAdaNorm – {ds_name}", plot_path)
        print("Saved figure →", plot_path)

    # Write JSON summary for this experiment.
    json_path = RESULTS_DIR / "experiment1_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print("==== Experiment 1 summary ====")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_exp1()
