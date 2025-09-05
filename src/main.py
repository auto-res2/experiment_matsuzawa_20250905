from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import yaml

# Local modules -------------------------------------------------------------
from .preprocess import FIG_DIR, RUN_DIR, set_seed, load_dataset
from .train import build_model, train
from .evaluate import save_line_plot

# ---------------------------------------------------------------------------
# Load external configuration (config/config.yaml)
# ---------------------------------------------------------------------------
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CFG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found: {CFG_PATH}")

with CFG_PATH.open("r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh)

# ---------------------------------------------------------------------------
# Experiment-1 orchestration (depth scaling)
# ---------------------------------------------------------------------------

def run_experiment_1() -> None:
    cfg_g = SimpleNamespace(**CONFIG["global"])

    print(
        """
===========================================================================
EXPERIMENT 1 – Depth-Scaling Benchmark
Datasets : {datasets}
Models   : {models}
Depths   : {depths}
===========================================================================""".format(
            datasets=", ".join(CONFIG["experiment_1"]["datasets"]),
            models=", ".join(CONFIG["experiment_1"]["models"]),
            depths=", ".join(map(str, CONFIG["experiment_1"]["depths"])),
        ),
        flush=True,
    )

    all_results: Dict[str, Dict[str, Dict[int, Dict]]] = {}

    for dname in CONFIG["experiment_1"]["datasets"]:
        print(f"\n>>> Loading dataset {dname} …", flush=True)
        data, num_cls = load_dataset(dname)
        in_dim = data.x.size(-1)
        dataset_res: Dict[str, Dict[int, Dict]] = {}

        for model_name in CONFIG["experiment_1"]["models"]:
            model_res: Dict[int, Dict] = {}
            for depth in CONFIG["experiment_1"]["depths"]:
                print(f"  • {model_name:<9s} L={depth:<3d}", flush=True)
                depth_res_acc: List[float] = []
                for seed in cfg_g.seeds:
                    set_seed(seed)
                    model = build_model(model_name, in_dim, cfg_g.hidden_dim, num_cls, depth)
                    stats = train(model, data, cfg_g)
                    depth_res_acc.append(stats["test_acc"])
                # mean across seeds
                mean_acc = sum(depth_res_acc) / len(depth_res_acc)
                stats["test_acc"] = mean_acc  # replace with mean value
                model_res[depth] = stats
                print(
                    f"    TestAcc={mean_acc:.3f}  APD={stats['apd']:.3f}  Reff={stats['reff']:.1f}  Time={stats['time_s'] / 60:.1f}m",
                    flush=True,
                )
            dataset_res[model_name] = model_res

            # Plot ----------------------------------------------------------
            xs = sorted(model_res.keys())
            ys = [model_res[d]["test_acc"] for d in xs]
            fname = f"accuracy_{dname}_{model_name}.pdf"
            save_line_plot(xs, ys, f"{model_name} on {dname}", "Test accuracy", fname)
            print(f"    Saved figure {fname}")
        all_results[dname] = dataset_res

    # Persist raw JSON for transparency ------------------------------------
    json_path = RUN_DIR / "experiment1_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nAll numerical results written to {json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – entry point
    cfg_snapshot = RUN_DIR / f"config_snapshot_{int(time.time())}.yaml"
    with cfg_snapshot.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(CONFIG, fh)
    print(f"Configuration snapshot saved to {cfg_snapshot}")

    run_experiment_1()


if __name__ == "__main__":
    main()
