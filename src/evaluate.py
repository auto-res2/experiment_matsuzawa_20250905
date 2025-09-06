"""src/evaluate.py – utilities for logging, JSON serialisation & plotting."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # head-less CI environments
import matplotlib.pyplot as plt


_RESEARCH_ROOT = Path(".research") / "iteration52"
_RESEARCH_ROOT.mkdir(parents=True, exist_ok=True)


class ExperimentBase:
    """Light-weight helper that manages directories + pretty printing."""

    def __init__(self, exp_id: str, exp_cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        self.exp_id = exp_id
        self.exp_cfg = exp_cfg
        self.global_cfg = global_cfg
        self.out_dir = _RESEARCH_ROOT / exp_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Experiment {exp_id} → output dir {self.out_dir}")

    # ---------------------------------------------------------------------
    def log_and_save(self, seed: int, result: Dict[str, Any]):
        fn = self.out_dir / f"results_seed{seed}.json"
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with fn.open("w", encoding="utf-8") as fp:
                json.dump(result, fp, indent=2)
        except Exception as exc:  # pragma: no cover
            print(f"[Err] Could not write {fn}: {exc}")
            raise
        print(f"[JSON] {fn.relative_to(Path('.'))} =\n{json.dumps(result, indent=2)}")

    # ---------------------------------------------------------------------
    def save_line_plot(self, y: List[float], x: List[int], ylabel: str, filename: str):
        try:
            plt.figure(figsize=(4, 3))
            plt.plot(x, y, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.tight_layout()
            path = self.out_dir / filename
            plt.savefig(path)
            plt.close()
            print(f"[Fig ] saved → {path.relative_to(Path('.'))}")
        except Exception as exc:
            print(f"[Warn] Could not save figure {filename}: {exc}")
