from __future__ import annotations

"""
src/main.py – experiment orchestrator
Run with:   python -m src.main
"""

import os
from pathlib import Path
from typing import Dict

import torch
import yaml

from .preprocess import build_stream
from .evaluate import MetricRecorder
from .train import CLoVeSub, ERRing, EWC, SparCL, TinyAQM

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

METHOD_FACTORY = {
    "clove_sub": CLoVeSub,
    "er_ring": ERRing,
    "ewc": EWC,
    "sparcl": SparCL,
    "aqm": TinyAQM,
}


def _load_yaml(path: Path) -> Dict:
    """Safely load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _find_default_cfg() -> Path:
    """Return the path to the default experiment YAML.

    Historically the repo expected a file named ``default.yaml`` but the
    template ships with ``config.yaml``.  We search for both so that either
    naming convention works out-of-the-box and users are free to rename their
    config file without touching any code.
    """
    for candidate in (CFG_DIR / "default.yaml", CFG_DIR / "config.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No experiment configuration file found – expected 'default.yaml' or 'config.yaml' inside the 'config/' folder."
    )


def _run_one_experiment(exp_cfg: Dict):
    print("\n====================  EXPERIMENT  ====================")
    print(exp_cfg.get("description", ""))
    print("=====================================================\n")

    # Build class-incremental streams
    train_s, val_s, test_s = build_stream(exp_cfg["dataset"])

    # run every method & seed
    for method_key, method_cfg in exp_cfg["methods"].items():
        ModelCls = METHOD_FACTORY[method_key]
        for seed in exp_cfg["seeds"]:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            model = ModelCls(exp_cfg, method_cfg).cuda()
            recorder = MetricRecorder(exp_cfg, method_key, seed)

            for task_id, (tr, va, te) in enumerate(zip(train_s, val_s, test_s)):
                model.before_task(task_id)
                model.train_task(task_id, tr, va, recorder)
                model.after_task(task_id)
                model.evaluate(task_id, te, recorder)

            recorder.close()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg_path = _find_default_cfg()
    cfg = _load_yaml(cfg_path)
    for exp in cfg["experiments"]:
        _run_one_experiment(exp)
