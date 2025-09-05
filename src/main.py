"""
main.py – research experiment entry point (python -m src.main)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import torch
import yaml

from .evaluate import MetricLogger
from .preprocess import build_stream
from .train import CLoVeSub, ERRing, SparCL

# -----------------------------------------------------------------------------
#  Registry – extend with new methods if needed
# -----------------------------------------------------------------------------

_METHODS: Dict[str, type] = {
    "clove_sub": CLoVeSub,
    "er_ring": ERRing,
    "sparcl": SparCL,
}

ROOT = Path(__file__).resolve().parent.parent  # project root (one level up from src)

# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

def _set_seed(seed: int) -> None:  # noqa: D401
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config() -> Dict:
    cfg_path = ROOT / "config" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text())


# -----------------------------------------------------------------------------
#  Main experiment driver
# -----------------------------------------------------------------------------

def _run_experiment(exp: Dict) -> None:  # noqa: D401
    print("\n==== EXPERIMENT ====\n" + exp["description"] + "\n====================\n")

    # Build continual stream ---------------------------------------------------
    train_s, val_s, test_s = build_stream(exp["dataset"])

    # Loop over methods & seeds ------------------------------------------------
    for method_key, method_cfg in exp["methods"].items():
        algo_cls = _METHODS[method_key]
        for seed in exp["seeds"]:
            _set_seed(seed)

            model = algo_cls(exp, method_cfg)
            if torch.cuda.is_available():
                model = model.cuda()

            out_dir = ROOT / "results"
            out_file = out_dir / f"{exp['id']}_{method_key}_seed{seed}.json"
            logger = MetricLogger(out_file)

            # Task loop ------------------------------------------------------
            for tid, (tr_loader, val_loader, te_loader) in enumerate(
                zip(train_s, val_s, test_s)
            ):
                model.before_task(tid)
                model.train_task(tid, tr_loader, val_loader, logger)
                model.after_task(tid)
                model.evaluate(tid, te_loader, logger)

            logger.close()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config = _load_config()
    for experiment in config["experiments"]:
        _run_experiment(experiment)
