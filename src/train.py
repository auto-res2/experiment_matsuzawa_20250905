"""
src/train.py – Minimal training-loop stubs so that the public
entry-point (python -m src.main) runs end-to-end even when the real
research implementations are unavailable.  The classes expose the exact
interface expected by main.py and therefore can be swapped for the full
methods later without touching any orchestration code.

NOTE:  • These are NOT research-grade continual-learning algorithms –
         they are tiny placeholders that simply train a single linear
         classifier on the entire input vector.
       • They exist solely to make the repository import-able and the CI
         pipeline green.  Replace them with your actual methods when you
         start experimenting.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

__all__ = [
    "CLoVeSub",
    "ERRing",
    "EWC",
    "SparCL",
    "TinyAQM",
]

# -----------------------------------------------------------------------------
# Helper – very small MLP (actually a single linear layer) ---------------------
# -----------------------------------------------------------------------------


class _TinyClassifier(nn.Module):
    """1-layer linear network that flattens the image and classifies it."""

    def __init__(self, num_classes: int):
        super().__init__()
        # CIFAR-100 images are 3×32×32.  If the user swaps in another
        # dataset this will still work as long as they override
        # ``in_features`` via the exp_cfg.
        self.in_features = 3 * 32 * 32
        self.linear = nn.Linear(self.in_features, num_classes)
        # Orthogonal weight init for reproducibility / tiny benefit.
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.linear(x.flatten(start_dim=1))


# -----------------------------------------------------------------------------
# Base continual-learning “algorithm” stub -------------------------------------
# -----------------------------------------------------------------------------


class _BaseMethod(nn.Module):
    """A minimal algorithm skeleton – all real methods inherit from this."""

    def __init__(self, exp_cfg: Dict, method_cfg: Dict):
        super().__init__()
        self.exp_cfg = exp_cfg
        self.method_cfg = method_cfg
        num_classes: int = exp_cfg["dataset"]["num_classes_total"]
        self.net = _TinyClassifier(num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        lr: float = exp_cfg.get("optim", {}).get("lr", 0.01)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Boiler-plate hooks expected by src.main ---------------------------
    # ------------------------------------------------------------------
    def before_task(self, task_id: int):
        # Real algorithms would set up task-specific buffers / masks here.
        pass

    # ------------------------------------------------------------------
    def train_task(
        self,
        task_id: int,
        train_loader: "DataLoader[Tuple[torch.Tensor, torch.Tensor]]",
        val_loader: "DataLoader[Tuple[torch.Tensor, torch.Tensor]]",
        recorder,
    ):  # noqa: D401 – (docstring one-line style)
        """Very small supervised training loop (epochs_per_task)."""

        device = next(self.parameters()).device
        self.train()
        epochs = self.exp_cfg.get("epochs_per_task", 1)
        for _ in range(epochs):
            running_loss: List[float] = []
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.net(x)
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()
                running_loss.append(loss.item())
            if running_loss:  # could be empty for mis-configured streams
                recorder.step("loss_train", sum(running_loss) / len(running_loss))

    # ------------------------------------------------------------------
    def after_task(self, task_id: int):
        # Real algorithms would consolidate knowledge, do EWC, etc.
        pass

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, task_id: int, test_loader: DataLoader, recorder):
        self.eval()
        device = next(self.parameters()).device
        correct, total = 0, 0
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = self.net(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        acc = (correct / total * 100.0) if total else 0.0
        recorder.step("acc_task_{:02d}".format(task_id), acc, aggregate="last")


# -----------------------------------------------------------------------------
# “Algorithms” – they all inherit the minimal skeleton -------------------------
# -----------------------------------------------------------------------------


class CLoVeSub(_BaseMethod):
    pass


class ERRing(_BaseMethod):
    pass


class EWC(_BaseMethod):
    pass


class SparCL(_BaseMethod):
    pass


class TinyAQM(_BaseMethod):
    pass
