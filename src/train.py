"""src.train – model definition and training-specific utilities for Dynamic Few-Bit Distillation (DFBD).
All logic has been extracted from the original monolithic script and cleaned up so that
other modules can import the model via

    from src.train import DFBDModel
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import yaml

# -----------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------
# -----------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}.  Make sure to run `main.py` "
        "from the project root so that config/config.yaml is generated or copied first."
    )
with CONFIG_PATH.open("r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

# -----------------------------------------------------------------------------
# Helper -----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _get_feature_extractor() -> Tuple[nn.Module, int]:
    """Return a ResNet-18 backbone truncated to the penultimate layer.
    The final 512-D feature vector is projected down to 256-D as specified
    in the YAML config.
    """
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the final fully-connected layer.
    backbone_layers = list(backbone.children())[:-1]  # Global average-pool output → (B, 512, 1, 1)
    feature_extractor = nn.Sequential(*backbone_layers, nn.Flatten())  # (B, 512)
    out_dim = 512
    proj_dim = CONFIG["models"]["backbone_imagenet"]["feature_dim"]
    projection = nn.Linear(out_dim, proj_dim)
    return nn.Sequential(feature_extractor, projection), proj_dim

# -----------------------------------------------------------------------------
# Core components --------------------------------------------------------------
# -----------------------------------------------------------------------------

class CoeffGenerator(nn.Module):
    """Small MLP Φ that maps feature vectors h→c in ℝ^K (later quantised to 8-bit)."""

    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True), nn.Linear(512, K)
        )
        self.K = K

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, in_dim) → (B, K)
        coeff = torch.tanh(self.mlp(h)) * 127  # scale to 8-bit range (-127 … 127)
        return coeff


class DFBDModel(nn.Module):
    """Minimal, self-contained DFBD implementation suitable for quick experiments."""

    def __init__(
        self,
        *,
        K: int = 48,
        rho: float = 0.002,
        bits_per_basis: int = 8,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.K = K
        self.rho = rho
        self.bits_per_basis = bits_per_basis

        # Backbone encoder
        self.encoder, feat_dim = _get_feature_extractor()
        # Linear classifier
        self.classifier = nn.Linear(feat_dim, num_classes)
        # Global basis (learned, float32 during optimisation – quantised only when
        # computing memory usage)
        self.basis = nn.Parameter(torch.randn(K, feat_dim))
        # Coefficient generator Φ
        self.coeff_gen = CoeffGenerator(feat_dim, K)

        # Replay buffers (kept on CPU to avoid unnecessary GPU RAM usage)
        self.register_buffer("coeff_buffer", torch.empty(0, K))  # (N, K)
        self.labels_buffer: List[int] = []

        self.samples_seen = 0
        self._compress_interval = CONFIG["models"]["dfbd"]["self_compress_interval"]

    # ---------------------------------------------------------------------
    # Forward & loss -------------------------------------------------------
    # ---------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        logits = self.classifier(h)
        return logits, h

    def forward_and_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, h = self.forward(x)
        loss_main = F.cross_entropy(logits, y)

        # Possibly store coefficients for future replay (with probability ρ)
        self._maybe_store_coeff(h.detach(), y.detach())

        # Rehearsal from buffer ------------------------------------------------
        if self.coeff_buffer.numel() > 0:
            replay_feats = torch.matmul(self.coeff_buffer.float(), self.basis)  # (N_replay, D)
            replay_logits = self.classifier(replay_feats)
            replay_labels = torch.tensor(
                self.labels_buffer, device=x.device, dtype=torch.long
            )
            loss_replay = F.cross_entropy(replay_logits, replay_labels)
            return loss_main + loss_replay
        return loss_main

    # ---------------------------------------------------------------------
    # Memory management ----------------------------------------------------
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _maybe_store_coeff(self, h: torch.Tensor, y: torch.Tensor):
        """Randomly store coefficients w.r.t sparsity ρ."""
        bs = h.size(0)
        coeff = self.coeff_gen(h)  # (B, K)
        mask = torch.rand(bs, device=h.device) < self.rho
        if mask.any():
            selected = coeff[mask].cpu()  # keep on CPU
            self.coeff_buffer = torch.cat([self.coeff_buffer.cpu(), selected], dim=0)
            self.labels_buffer.extend(y[mask].cpu().tolist())
        self.samples_seen += bs

    def should_self_compress(self) -> bool:
        return self.samples_seen > 0 and self.samples_seen % self._compress_interval == 0

    # ---------------------------------------------------------------------
    # Self-compression ------------------------------------------------------
    # ---------------------------------------------------------------------

    def self_compress(self, *, epochs: int = 1, lr: float = 1e-2):
        """Merge stored coefficients into the basis to keep memory constant."""
        if self.coeff_buffer.numel() == 0:
            return  # Nothing to compress.

        opt = torch.optim.SGD([self.basis], lr=lr)
        dataset_coeff = self.coeff_buffer.float().to(self.basis.device)
        loader = torch.utils.data.DataLoader(dataset_coeff, batch_size=256, shuffle=True)

        for _ in range(epochs):
            for c in loader:
                opt.zero_grad()
                recon = torch.matmul(c, self.basis)  # (B, D)
                target = torch.matmul(c, self.basis.detach())  # detach to avoid trivial solution
                loss = (recon - target).pow(2).mean()
                loss.backward()
                opt.step()

        # Quantise & prune ------------------------------------------------
        self._quantise_basis()
        self._prune_coeffs()

    # ---------------------------------------------------------------------
    # Helpers --------------------------------------------------------------
    # ---------------------------------------------------------------------

    def _quantise_basis(self):
        with torch.no_grad():
            max_abs = self.basis.abs().max().clamp(min=1e-5)
            scale = 127 / max_abs
            q = torch.round(self.basis * scale).to(torch.int8)
            self.basis.data = (q.float() / scale)

    def _prune_coeffs(self):
        with torch.no_grad():
            magn = self.coeff_buffer.abs().sum(dim=1)  # (N,)
            keep = magn > 1e-3
            if keep.any():
                self.coeff_buffer = self.coeff_buffer[keep]
                self.labels_buffer = [lbl for lbl, k in zip(self.labels_buffer, keep.tolist()) if k]
            else:
                # If nothing is kept, clear completely to avoid shape mismatches later.
                self.coeff_buffer = torch.empty(0, self.K, device=self.coeff_buffer.device)
                self.labels_buffer = []

    # ---------------------------------------------------------------------
    # Reporting ------------------------------------------------------------
    # ---------------------------------------------------------------------

    def byte_size(self) -> int:
        """Return total persistent memory in **bytes** (basis + 8-bit coeffs)."""
        basis_bytes = self.basis.numel() * (self.bits_per_basis // 8)
        coeff_bytes = self.coeff_buffer.numel()  # each int8 → 1 byte
        return int(basis_bytes + coeff_bytes)


__all__ = [
    "DFBDModel",
]
