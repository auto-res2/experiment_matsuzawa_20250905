"""
src/evaluate.py – Metric recorder, FLOPs + energy profiler and evaluation helpers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# -------- torch profiler --------
try:
    from torch.profiler import ProfilerActivity, profile, schedule
except Exception:  # pragma: no cover
    profile = None  # type: ignore

# -------- NVML energy --------
try:
    import pynvml

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover
    _NVML_AVAILABLE = False


class MetricRecorder:
    """Light-weight recorder that tracks metrics, energy and FLOPs."""

    def __init__(self, exp_cfg: Dict, method: str, seed: int):
        self.data: Dict[str, List[float]] = {}
        self.out_path = Path("results") / f"{exp_cfg['id']}_{method}_{seed}.json"
        self.out_path.parent.mkdir(exist_ok=True, parents=True)

        # torch.profiler (optional)
        if profile is not None:
            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
                with_flops=True,
            )
            self.prof.__enter__()
        else:
            self.prof = None

        # NVML energy (optional)
        if _NVML_AVAILABLE:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.energy_prev = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        else:
            self.handle = None
            self.energy_prev = 0

    # ------------------------------------------------------------------
    def step(self, key: str, val: float, aggregate: str = "mean"):
        if key not in self.data:
            self.data[key] = []
        if aggregate == "last":
            self.data[key] = [val]
        else:
            self.data[key].append(val)
        if self.prof is not None:
            self.prof.step()

    # ------------------------------------------------------------------
    def close(self):
        # profiler
        if self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.data["flops"] = self.prof.key_averages().total_average().flops
        else:
            self.data["flops"] = 0.0

        # energy
        if _NVML_AVAILABLE and self.handle is not None:
            energy_now = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.data["energy_J"] = (energy_now - self.energy_prev) / 1e3
        else:
            self.data["energy_J"] = 0.0

        # persist
        with self.out_path.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)
        print(f"[metrics saved] {self.out_path}")
