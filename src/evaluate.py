from __future__ import annotations

"""
evaluate.py – generic helper utilities for logging, plotting and storing
per-experiment JSON results.  No experiment-specific code lives here.

Key change (iteration8):
  • All results and images must now reside under the directory
        .research/iteration8/
    as required by the latest assessment instructions.
"""

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib

# Use a non-interactive backend because the code may run on a headless CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  – after backend selection
import seaborn as sns  # noqa: E402

# ---------------------------------------------------------------------------
#                      OPTIONAL  GPU  UTILISATION  LOGGING
# ---------------------------------------------------------------------------
# We try to import NVML bindings.  If the import (or any subsequent call)
# fails we fall back to a dummy implementation – this must never crash the
# experiment.
try:
    from pynvml import (  # type: ignore
        nvmlInit,  # type: ignore
        nvmlDeviceGetHandleByIndex,  # type: ignore
        nvmlDeviceGetUtilizationRates,  # type: ignore
    )

    nvmlInit()
    _NVML_HANDLE = nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover – best-effort only
    _NVML_HANDLE = None  # type: ignore

__all__ = ["ExperimentBase"]

# ---------------------------------------------------------------------------
# Directories mandated by the assessment instructions (iteration8)
# ---------------------------------------------------------------------------
_BASE_RESULTS_DIR = Path(".research/iteration8")
_IMAGES_DIR = _BASE_RESULTS_DIR / "images"
_BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class ExperimentBase:
    """Superclass that provides JSON logging and plotting utilities."""

    def __init__(self, name: str, exp_cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        self.name = name
        self.cfg = exp_cfg
        self.global_cfg = global_cfg
        # Keep a sub-directory for any auxiliary artefacts the experiment wants
        # to dump (e.g. counterfactual samples) but store *results* & *figures*
        # strictly under .research/iteration8/ as required by the rubric.
        self.results_dir = _BASE_RESULTS_DIR / name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _gpu_util() -> float | None:
        if _NVML_HANDLE is None:
            return None
        util = nvmlDeviceGetUtilizationRates(_NVML_HANDLE)  # type: ignore[arg-type]
        # type: ignore[return-value]
        return float(util.gpu)  # pyright: ignore[reportOptionalMemberAccess]

    # ------------------------------------------------------------------
    def log_and_save(self, seed: int, result: Dict[str, Any]):
        """Save *result* as JSON and print it to STDOUT for verification."""
        json_path = _BASE_RESULTS_DIR / f"{self.name}_seed{seed}.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)

        # STDOUT – human-readable header followed by the full JSON blob
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
        """Utility that writes a small line plot under .research/iteration8/images."""
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=list(xs), y=list(ys), marker="o", label=ylabel)
        for x_val, y_val in zip(xs, ys):
            plt.text(x_val, y_val, f"{y_val:.1f}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(_IMAGES_DIR / fig_name, bbox_inches="tight")
        plt.close()
