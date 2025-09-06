"""src/train.py
-------------------------------------------------------------------------------
Training-related classes and functions that are used by the public entry-point
``python -m src.main``.  Only model building, continual-learning memory and the
actual optimisation step live in this module so they can be unit-tested
independently from data-loading and evaluation utilities.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 1.  LATENT ORTHOGONAL SKETCH REPLAY (LOSR)  – memory implementation
# -----------------------------------------------------------------------------

class AnchorBank:
    """Maintains per-class anchors μ_c (EMA) and top-r orthogonal directions U_c.

    Memory cost in bytes can be queried via ``bytes()`` – this is an *upper
    bound* because we keep them in FP32 for numerical stability even though the
    paper quantises them.
    """

    def __init__(self, feat_dim: int, r: int = 2, ema_momentum: float = 0.01):
        self.feat_dim = feat_dim
        self.r = r
        self.mom = ema_momentum
        self.mu: Dict[int, torch.Tensor] = {}
        self.U: Dict[int, torch.Tensor] = {}
        self.cov: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """Streaming update given *L2-normalised* feature vectors and labels."""
        for z, y in zip(feats, labels):
            y = int(y)
            if y not in self.mu:
                self.mu[y] = z.clone()
                self.cov[y] = torch.zeros(self.feat_dim, device=z.device)
                self.U[y] = torch.eye(self.feat_dim, device=z.device)[:, : self.r]
            else:
                # ----- exponential moving average for anchor μ
                self.mu[y].mul_(1 - self.mom).add_(z * self.mom)
                # ----- cheap diagonal covariance sketch for principal directions
                delta = z - self.mu[y]
                self.cov[y] += delta * delta
                top_idx = torch.topk(self.cov[y], k=self.r).indices
                u = torch.zeros(self.feat_dim, self.r, device=z.device)
                for k, idx in enumerate(top_idx):
                    u[idx, k] = 1.0
                self.U[y] = F.normalize(u, dim=0)

    def sample(self, class_id: int, n: int = 64) -> torch.Tensor:
        """Draw *n* synthetic latent vectors for given class ID."""
        mu = self.mu[class_id]
        U = self.U[class_id]
        eps = torch.randn(n, self.r, device=mu.device)
        z = mu + (eps @ U.T)
        return F.normalize(z, dim=1)

    # ---------------- memory footprint -------------------------------------------------
    def bytes(self) -> int:
        total = 0
        for c in self.mu:
            total += self.mu[c].numel() + self.U[c].numel()
        return int(total * 4)  # FP32 bytes per element


class FeatureSynthesiser(nn.Module):
    """Lightweight hyper-network g_φ:  (ε, class-id) -> latent feature ẑ."""

    def __init__(self, feat_dim: int, n_classes_max: int, noise_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_classes_max, 16)
        self.fc1 = nn.Linear(noise_dim + 16, 128)
        self.fc2 = nn.Linear(128, feat_dim)

    def forward(self, noise: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = torch.cat([noise, self.embed(cls)], dim=1)
        h = F.relu(self.fc1(h))
        return F.normalize(self.fc2(h), dim=1)


class LOSRMemory(nn.Module):
    """Wrapper exposing *generate* and *update_bank* like Avalanche ReplayPlugin."""

    def __init__(self, feat_dim: int, r: int = 2, budget_kb: int = 32, n_classes_max: int = 100):
        super().__init__()
        self.bank = AnchorBank(feat_dim, r)
        self.synth = FeatureSynthesiser(feat_dim, n_classes_max)
        self.budget_kb = budget_kb

    @torch.no_grad()
    def generate(self, n_per_class: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.bank.mu:
            raise ValueError("No anchors yet – cannot sample.")
        feats, labels = [], []
        for cls in self.bank.mu:
            f = self.bank.sample(cls, n_per_class)
            feats.append(f)
            labels.extend([cls] * n_per_class)
        return torch.cat(feats), torch.tensor(labels, device=feats[0].device)

    def forward(self, noise: torch.Tensor, cls: torch.Tensor):  # proxy
        return self.synth(noise, cls)

    # ---------------- interface for trainer -------------------------------------------
    def update_bank(self, feats: torch.Tensor, labels: torch.Tensor):
        self.bank.update(feats, labels)

    def bytes(self) -> int:
        return self.bank.bytes()


# -----------------------------------------------------------------------------
# 2.  BACKBONE + CLASSIFIER UTILITIES
# -----------------------------------------------------------------------------

def build_backbone(name: str) -> nn.Module:
    """Return normalised 256-D feature extractor without task-specific head."""
    if name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Identity()
        proj = nn.Linear(512, 256)
        backbone = nn.Sequential(model, proj, nn.LayerNorm(256, elementwise_affine=False))
    elif name == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(weights=None)
        model.classifier = nn.Identity()
        proj = nn.Linear(model.last_channel, 256)
        backbone = nn.Sequential(model, proj, nn.LayerNorm(256, elementwise_affine=False))
    else:
        raise ValueError(name)
    return backbone


class ExpandingClassifier(nn.Module):
    """Linear classifier that automatically grows when new classes appear."""

    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 0
        # Start with *no* output neurons.  We will grow the layer on-the-fly as
        # new classes become visible.
        self.fc = nn.Linear(in_dim, 0, bias=True)

    # ---------------------------------------------------------------------
    # Public helper --------------------------------------------------------

    def ensure_capacity(self, n_classes: int) -> None:
        """Grow the classifier so that it can predict ``n_classes`` classes.

        This helper is *idempotent* -- calling it with a number smaller than or
        equal to ``self.out_dim`` is a no-op.
        """
        if n_classes > self.out_dim:
            self.add_classes(n_classes - self.out_dim)

    # ---------------------------------------------------------------------
    # Internal – actually perform the layer surgery -----------------------

    def add_classes(self, n: int):
        """Physically append ``n`` output neurons to the linear layer."""
        weight_old = self.fc.weight.data
        bias_old = self.fc.bias.data if self.fc.bias is not None else None
        new_fc = nn.Linear(self.in_dim, self.out_dim + n, bias=True)
        # Copy over existing parameters ------------------------------------------------
        if self.out_dim:
            new_fc.weight.data[: self.out_dim] = weight_old
            new_fc.bias.data[: self.out_dim] = bias_old
        # Move to the same device as the old layer -------------------------------------
        new_fc = new_fc.to(self.fc.weight.device)
        self.fc = new_fc
        self.out_dim += n

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return self.fc(x)


# -----------------------------------------------------------------------------
# 3.  SINGLE-EXPERIENCE TRAINING STEP
# -----------------------------------------------------------------------------

def train_one_experience(
    backbone: nn.Module,
    clf: ExpandingClassifier,
    losr: LOSRMemory | None,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
):
    """Streams through one experience (task) and optionally performs LOSR replay."""

    backbone.train()
    clf.train()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # ------------------------------------------------------------------
        # 1)  Make sure the classifier can handle the *real* batch labels.
        # ------------------------------------------------------------------
        clf.ensure_capacity(int(y.max().item()) + 1)

        optimiser.zero_grad()

        feats = backbone(x)
        logits = clf(feats)
        loss = F.cross_entropy(logits, y)

        # ------------------------------------------------------------------
        # 2)  Optional LOSR replay.
        # ------------------------------------------------------------------
        if losr is not None and losr.bank.mu:
            syn_feats, syn_labels = losr.generate(n_per_class=16)
            # Some classes might *just* have been added; double-check capacity.
            clf.ensure_capacity(int(syn_labels.max().item()) + 1)

            loss = loss + F.cross_entropy(clf(syn_feats), syn_labels)

            noise = torch.randn(len(syn_labels), 8, device=device)
            gen_feats = losr(noise, syn_labels)
            loss = loss + F.cross_entropy(clf(gen_feats), syn_labels)

        # ------------------------------------------------------------------
        # 3)  Optimise.
        # ------------------------------------------------------------------
        loss.backward()
        optimiser.step()

        # ------------------------------------------------------------------
        # 4)  Update anchor bank *after* weights were updated so that the new
        #     features reflect the current state of the backbone.
        # ------------------------------------------------------------------
        if losr is not None:
            losr.update_bank(feats.detach(), y.detach())
