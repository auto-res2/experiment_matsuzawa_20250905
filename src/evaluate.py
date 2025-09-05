"""src/evaluate.py
Functions for sanity-checking, metric logging, statistical analysis and
plotting.  They do *not* depend on the actual data pipeline and can
therefore be imported prior to downloading datasets (fast CI checks).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch

try:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML = True
except Exception:  # pragma: no cover – NVML not available in CPU CI
    _NVML = False
    NVML_HANDLE = None  # type: ignore


# ----------------------------------------------------------------------
#  Metric logger – JSON + (optionally) energy via NVML
# ----------------------------------------------------------------------


class MetricLogger:
    def __init__(self, out_file: Path):
        self.data: Dict[str, List[float]] = {}
        self.file = out_file
        self.t0 = time.time()
        out_file.parent.mkdir(exist_ok=True, parents=True)

    def log(self, key: str, value: float):
        self.data.setdefault(key, []).append(float(value))

    def close(self):
        self.data["wall_clock"] = time.time() - self.t0
        if _NVML:
            self.data["energy_J"] = (
                pynvml.nvmlDeviceGetTotalEnergyConsumption(NVML_HANDLE) / 1000.0
            )
        with open(self.file, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"[MetricLogger] Saved to {self.file}")


# ----------------------------------------------------------------------
#  Stage-0 sanity gate used before every run
# ----------------------------------------------------------------------


def stage0_sanity(model, config):  # noqa: ANN001
    """Light-weight unit tests executed before starting expensive training."""

    print("[Stage-0] Fidelity & Sanity Gate running …")
    device = next(model.parameters()).device

    # Orthogonality of the projector -------------------------------------
    with torch.no_grad():
        ortho_err = (
            model.projector.P.t() @ model.projector.P
            - torch.eye(model.projector.P.shape[1], device=device)
        ).norm().item()
    assert ortho_err < 1e-4, f"Projector not orthogonal (‖PtP−I‖={ortho_err})"

    # Encoder/quantiser reconstruction error -----------------------------
    xb = torch.randn(
        32,
        3,
        config["dataset"]["img_size"],
        config["dataset"]["img_size"],
        device=device,
    )
    with torch.no_grad():
        z_e = model.encoder(xb)
        z_q, _, _ = model.vq(z_e, tau=config["models"]["vq"]["tau"])
        mse = torch.nn.functional.mse_loss(z_e, z_q).item()
    assert mse < 0.03, f"VQ recon MSE too high ({mse})"

    # Latent adapter agreement ------------------------------------------
    codes = torch.randint(
        0,
        config["models"]["vq"]["n_codes"],
        (
            32,
            config["dataset"]["img_size"] // 4,
            config["dataset"]["img_size"] // 4,
        ),
        dtype=torch.uint8,
        device=device,
    )
    with torch.no_grad():
        feat_pred = model.adapter(codes, model.vq.codebook)
    assert feat_pred.shape[1] == 512, "Adapter output has wrong feature dim"

    # Quick timing & energy sanity --------------------------------------
    if _NVML:
        energy0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(NVML_HANDLE)
    t0 = time.time()
    for _ in range(5):
        _ = model.forward(torch.randn(4, 3, 32, 32, device=device).float())
    wall = time.time() - t0
    if _NVML:
        energy1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(NVML_HANDLE)
        assert energy1 > energy0, "NVML energy failed to increase"
    assert wall > 0, "Wall-clock timing failed"
    print("[Stage-0] All sanity checks passed.\n")


# ----------------------------------------------------------------------
#  Plotting helper – returns figure handle so that caller can save
# ----------------------------------------------------------------------


def plot_accuracy(per_task_acc, out_file: Path):  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(per_task_acc)), [v * 100 for v in per_task_acc], marker="o")
    for i, v in enumerate(per_task_acc):
        ax.annotate(f"{v*100:.1f}", (i, v * 100 + 0.5), fontsize=6)
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Continual Accuracy per Task – CLoVe-Sub")
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy plot to {out_file}")
