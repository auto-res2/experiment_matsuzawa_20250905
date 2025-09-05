"""
evaluate.py – utilities for logging, statistical analysis & plotting stubs
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

try:
    # Optional – only if the machine has NVML (e.g. an NVIDIA GPU)
    import pynvml  # type: ignore

    pynvml.nvmlInit()

    _NVML_AVAILABLE = True
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover – safe fallback for CPU boxes
    _NVML_AVAILABLE = False
    _HANDLE = None  # type: ignore


class MetricLogger:
    """Very light JSON logger used during training / evaluation."""

    def __init__(self, out_file: Path):
        self._data: Dict[str, List[float]] = {}
        self._t0 = time.time()
        self._file = out_file
        self._file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def log(self, key: str, value: float) -> None:  # noqa: D401
        self._data.setdefault(key, []).append(float(value))

    # ------------------------------------------------------------------
    def close(self) -> None:  # noqa: D401
        self._data["wall_clock_sec"] = time.time() - self._t0
        if _NVML_AVAILABLE:
            # Energy consumption in millijoules → joules
            self._data["energy_J"] = (
                pynvml.nvmlDeviceGetTotalEnergyConsumption(_HANDLE) / 1_000.0
            )
        with self._file.open("w") as f:
            json.dump(self._data, f, indent=2)
        print(f"[MetricLogger] Saved metrics to {self._file}")


# Note: full statistical analysis & plotting helpers are not required for the
# refactor demo.  They would be added here in a real research code-base.
