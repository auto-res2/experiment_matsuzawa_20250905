"""src/train.py – model definition and training utilities for CLoVe-Sub experiments
-------------------------------------------------------------------------------
The original monolithic script has been refactored: everything that is needed to
create the model and perform one epoch/batch of optimisation lives here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

__all__ = [
    "SimpleCloVeSubNet",
    "training_loop",
]


class SimpleCloVeSubNet(nn.Module):
    """A *compact* surrogate for the full CLoVe-Sub architecture.

    The full method would include VQ-VAE encoders, latent adapters, orthogonal
    projectors, etc.  For the purpose of running the public experiment end-to-
    end we only need a backbone and a multi-head classifier.  The interface is
    identical to the original single-file implementation so the experiment
    driver remains unchanged.
    """

    def __init__(
        self,
        feat_dim: int = 512,
        num_tasks: int = 50,
        classes_per_task: int = 2,
    ) -> None:
        super().__init__()

        self.backbone = torchvision.models.resnet18(weights=None)  # PyTorch ≥2.1 API
        # Replace the final fully-connected layer by an identity mapping so we
        # can attach task-specific heads.
        self.backbone.fc = nn.Identity()
        self.feat_dim = feat_dim

        # One linear head per task (a *multi-head* continual-learning setup).
        self.heads = nn.ModuleList(
            [nn.Linear(feat_dim, classes_per_task) for _ in range(num_tasks)]
        )

    def forward(self, x: torch.Tensor, task_id: int):  # noqa: D401
        """Forward pass for a single task.

        Parameters
        ----------
        x : torch.Tensor
            The input mini-batch of images.
        task_id : int
            Which task-specific classifier head to use.
        """
        feat = self.backbone(x)
        return self.heads[task_id](feat)


# -----------------------------------------------------------------------------
#  Optimisation / training utilities
# -----------------------------------------------------------------------------

def training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    task_id: int,
    epochs: int = 1,
) -> None:
    """Standard supervised training loop with cross-entropy loss."""

    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images, task_id)
            loss = F.cross_entropy(logits, labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()