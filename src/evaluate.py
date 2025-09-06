"""
evaluate.py – generic helper utilities for logging, plotting and storing
per-experiment JSON results.  No experiment-specific code lives here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# optional – only for GPU utilisation logging; failing gracefully is OK
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates

    nvmlInit()
    _NVML_HANDLE = nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover – best effort only
    _NVML_HANDLE = None  # type: ignore

__all__ = ["ExperimentBase"]


class ExperimentBase:
    """Superclass that provides JSON logging and plotting utilities."""

    def __init__(self, name: str, exp_cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        self.name = name
        self.cfg = exp_cfg
        self.global_cfg = global_cfg
        self.results_dir = Path(global_cfg["results_dir"]) / self.name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _gpu_util() -> float | None:
        if _NVML_HANDLE is None:
            return None
        util = nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
        return float(util.gpu)  # type: ignore

    # ------------------------------------------------------------------
    def log_and_save(self, seed: int, result: Dict[str, Any]):
        json_path = self.results_dir / f"seed{seed}.json"
        with open(json_path, "w") as fp:
            json.dump(result, fp, indent=2)

        # STDOUT – text description followed by full JSON
        print(f"\n=====  {self.name}  |  seed={seed}  =====")
        print(json.dumps(result, indent=2, sort_keys=True))

    # ------------------------------------------------------------------
    def save_line_plot(
        self,
        ys: Sequence[float],
        xs: Sequence[int],
        ylabel: str,
        fig_name: str,
    ) -> None:
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=list(xs), y=list(ys), marker="o", label=ylabel)
        for x_val, y_val in zip(xs, ys):
            plt.text(x_val, y_val, f"{y_val:.1f}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(self.results_dir / fig_name, bbox_inches="tight")
        plt.close()
