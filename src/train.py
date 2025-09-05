from __future__ import annotations

import hashlib
from typing import List

import torch
import torch.fft as fft
import torch.nn as nn
from torch.utils.data import DataLoader

__all__ = [
    "fourier_cd",
    "GroupDROTrainer",
]


def fourier_cd(x: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Fourier Consistent-Distance (CD) as in *Models Out-of-Line*.

    Parameters
    ----------
    x: torch.Tensor
        Image mini-batch in **[B×C×H×W]** format and *float* range [0,1].
    window: int, optional
        Size of the low-frequency square window kept un-penalised.

    Returns
    -------
    torch.Tensor (scalar)
        CD regularisation term – higher ⇒ stronger high-frequency usage.
    """
    f = fft.fftn(x.float(), dim=(-2, -1))
    f = fft.fftshift(f, dim=(-2, -1))
    h, w = x.shape[-2:]
    centre = f[..., h // 2 - window : h // 2 + window, w // 2 - window : w // 2 + window]
    low = centre.abs().pow(2).mean()
    total = f.abs().pow(2).mean()
    # Normalised high-frequency energy
    return ((total - low) / (total + 1e-8)).mean()


class GroupDROTrainer:
    """Classifier optimiser with Group-Conditional DRO + Fourier regulariser."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        dro_radius: float = 0.2,
        fourier_lambda: float = 0.2,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.dro_radius = dro_radius
        self.fourier_lambda = fourier_lambda

    # ---------------------------------------------------------------------
    #  Internal helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _hash_gid(batch_idx: int, sel_dim: int, sign: int) -> int:
        """Deterministic 64-bit hash used as *group id*."""
        h = hashlib.sha1(f"{batch_idx}_{sel_dim}_{sign}".encode()).digest()
        return int.from_bytes(h[:8], "big")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> None:
        self.model.train()
        for x, y, gid in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            gid = gid.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss_individual = self.loss_fn(logits, y)

            # Group-worst loss (Group DRO)
            uniq = torch.unique(gid)
            g_loss = torch.stack([loss_individual[gid == g].mean() for g in uniq])
            worst_group = uniq[g_loss.argmax()]
            loss = loss_individual[gid == worst_group].mean()

            # Fourier regularisation --------------------------------------------------
            if self.fourier_lambda > 0:
                loss = loss + self.fourier_lambda * fourier_cd(x)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

    @torch.no_grad()
    def evaluate_accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)
