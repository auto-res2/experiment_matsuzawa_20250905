from __future__ import annotations

import torch
from torchmetrics.classification import MulticlassCalibrationError
from torch.utils.data import DataLoader

__all__ = ["expected_calibration_error"]


def expected_calibration_error(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str | torch.device = "cuda",
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) computed with torchmetrics."""
    device = torch.device(device)
    ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins).to(device)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            ece_metric.update(logits.softmax(-1), y)
    return float(ece_metric.compute())
