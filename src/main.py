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
  4. Saves the resulting metrics as JSON under `.research/iteration4` **and**
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


def _pip_available() -> bool:
    """Return True iff the current interpreter has the `pip` module bundled."""
    import importlib.util

    return importlib.util.find_spec("pip") is not None


def _install_repo(name: str, url: str):
    """Attempt to import `name`; if that fails and pip is available, try to
    install from `url`.  If pip is missing or installation fails, we log a
    warning and continue – the experiment will only break later if that
    dependency is *actually* required.  This avoids hard crashes when the
    external baselines are *not* used (the common case for the provided YAML).
    """
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass  # not installed – try to fetch below

    if not _pip_available():
        print(f"[setup] pip unavailable – skipping installation of {name}.", flush=True)
        return

    try:
        print(f"[setup] Installing {name} from {url} …", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])
    except subprocess.CalledProcessError as e:
        print(f"[setup] Failed to install {name}: {e}. Proceeding without.", flush=True)


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
_RESULTS_DIR = Path(".research/iteration4")  # UPDATED PATH
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
