from __future__ import annotations
"""src/main.py ––– project entry-point (patched to use iteration11 layout)"""
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from .train import run_experiment
from .evaluate import generate_all_figures
from .utils import banner, set_global_seed

# ---------------------------------------------------------------------------
# 0.  Reproducibility
# ---------------------------------------------------------------------------
set_global_seed(42)
print(banner("Dynamic Halting GNN – Reproducible Experiment Suite"))

# ---------------------------------------------------------------------------
# 1.  Directory bootstrap (.research/iteration11/*)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / ".research" / "iteration11"
IMAGES_DIR = RESEARCH_DIR / "images"

for d in (RESEARCH_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Ensure local src/ is importable when users `python -m src.main`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 2.  Load YAML configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf8") as f:
    CFG: Dict[str, Any] = yaml.safe_load(f)

print(f"[INFO] Configuration loaded from {CONFIG_PATH}")

# ---------------------------------------------------------------------------
# 3.  Helpers
# ---------------------------------------------------------------------------

def _save_json(d: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(d, f, indent=2)


def main() -> None:
    all_results: Dict[str, Dict[str, Any]] = {}

    for exp_name, exp_cfg in CFG["experiments"].items():
        print("\n" + banner(f"Starting {exp_name}"))
        json_out_path = RESEARCH_DIR / f"{exp_name}.json"

        start = time.time()
        results = run_experiment(exp_name, exp_cfg, CFG)
        runtime = time.time() - start
        results["wall_clock_seconds"] = runtime

        _save_json(results, json_out_path)
        all_results[exp_name] = results

        # ---- CLI verification ------------------------------------------
        print("\n=====  EXPERIMENT DESCRIPTION  =====")
        print(textwrap.fill(exp_cfg["description"], width=100))

        print("\n=====  NUMERICAL RESULTS  =====")
        print(json.dumps(results, indent=2))
        print(f"[INFO] Individual result saved → {json_out_path}")

        # ---- figure generation -----------------------------------------
        generate_all_figures(exp_name, results, IMAGES_DIR)

    # -----------------------------------------------------------------
    # Combined JSON dump (checked in CI)
    # -----------------------------------------------------------------
    final_path = RESEARCH_DIR / "all_experiments.json"
    _save_json(all_results, final_path)

    print("\n" + banner("ALL EXPERIMENTS COMPLETE"))
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Run interrupted by user – exiting.")
