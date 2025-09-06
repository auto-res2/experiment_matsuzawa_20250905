"""
main.py
=======
Entry point for all experiments.  The script is executed as

    python -m src.main

It performs the following steps:
  1. Installs missing third-party baseline repos (PairNorm, NDLS, …) **before**
     importing PyTorch to avoid duplicate CUDA context creation.
  2. Reads the YAML configuration from `config/config.yaml` using PyYAML.
  3. Runs each experiment via `src.train.run_experiment`.
  4. Saves the resulting metrics as JSON under `.research/iteration1` **and**
     prints the JSON object to stdout – the evaluation harness depends on this.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import yaml

# ---------------------------------------------------------------------------
# 0.  Dynamically install 3rd-party baselines (PairNorm / NDLS / DGN)
# ---------------------------------------------------------------------------
_REPO_REQUIREMENTS: Dict[str, str] = {
    "PairNorm": "git+https://github.com/LingxiaoShawn/PairNorm.git",
    "NDLS": "git+https://github.com/zwt233/NDLS.git",
    "DGN": "git+https://github.com/Kaixiong-Zhou/DGN.git",
}


def _install_repo(name: str, url: str):
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"[setup] Installing {name} from {url} …", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])


for _lib, _url in _REPO_REQUIREMENTS.items():
    _install_repo(_lib, _url)

# ---------------------------------------------------------------------------
# Now safe to import torch / heavy libs
# ---------------------------------------------------------------------------
from .train import run_experiment

# ---------------------------------------------------------------------------
# Helper: pretty print JSON and guarantee flush (evaluation harness requires)
# ---------------------------------------------------------------------------

_JSON_OPTS = dict(indent=2, sort_keys=True)

def _echo_json(d: Dict):
    js = json.dumps(d, **_JSON_OPTS)
    print(js, flush=True)
    return js

# ---------------------------------------------------------------------------
# 1.  Load configuration tree (PyYAML instead of Hydra for simplicity)
# ---------------------------------------------------------------------------

_CFG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
if not _CFG_PATH.is_file():
    raise FileNotFoundError(f"Config not found at {_CFG_PATH}")
with _CFG_PATH.open("r") as fp:
    CFG = yaml.safe_load(fp)

# ---------------------------------------------------------------------------
# 2.  Run the experiments
# ---------------------------------------------------------------------------
_RESULTS_DIR = Path(".research/iteration1")
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    experiments = CFG.get("experiments", {})
    if not experiments:
        print("[WARN] No experiments found in YAML config.")
        return

    for exp_name, exp_cfg in experiments.items():
        desc = exp_cfg.get("description", "")
        print(f"\n========== Running {exp_name}: {desc} ==========")
        sys.stdout.flush()

        try:
            metrics, figs = run_experiment(exp_name, exp_cfg)
        except Exception as e:
            print(f"[ERROR] Experiment '{exp_name}' failed – {e}", file=sys.stderr)
            raise  # STRICT NO-FALLBACK RULE

        # ------------- persist & echo ----------------
        out_path = _RESULTS_DIR / f"{exp_name}.json"
        metrics["figures"] = figs
        with out_path.open("w") as fp:
            json.dump(metrics, fp, **_JSON_OPTS)
        print(f"\n[RESULT] {exp_name} → {out_path}")
        _echo_json(metrics)

    print("\nAll experiments finished successfully.")


if __name__ == "__main__":
    main()
