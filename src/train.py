"""
train.py – model architectures, buffers and training algorithms
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def sparsify_grads(model: nn.Module, theta: float = 0.2) -> None:
    """Keep top-(1-θ) fraction of the gradients by magnitude.

    A very small and fast variant of the SparCL mask used in the paper.
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.view(-1)
        k = int((1 - theta) * g.numel())
        if k <= 0:
            p.grad.zero_()
            continue
        thr = g.abs().kthvalue(k).values.item()
        mask = g.abs() >= thr
        p.grad.mul_(mask.view_as(p.grad))


# -----------------------------------------------------------------------------
#  Model building blocks
# -----------------------------------------------------------------------------

try:
    import geoopt  # type: ignore
except ImportError:  # pragma: no cover – geoopt is optional on CPU–only boxes
    geoopt = None


class StiefelProjector(nn.Module):
    """Linear projector whose weight lies on the Stiefel manifold (orthogonal).

    Keeps the last-layer features in a low-rank orthogonal sub-space in order
    to minimise interference across tasks.
    """

    def __init__(self, in_dim: int, rank: int = 16):
        super().__init__()

        # geoopt available  ➜ use proper manifold parameter
        if geoopt is not None:
            self.weight = geoopt.ManifoldParameter(  # type: ignore[attr-defined]
                torch.empty(in_dim, rank), manifold=geoopt.Stiefel()  # type: ignore[attr-defined]
            )
            nn.init.orthogonal_(self.weight)
            self._use_geoopt = True
        # geoopt NOT available  ➜ fall back to unconstrained parameter
        else:
            self.weight = nn.Parameter(torch.empty(in_dim, rank))
            nn.init.orthogonal_(self.weight)
            self.register_buffer("_warned", torch.tensor(0, dtype=torch.uint8), persistent=False)
            self._use_geoopt = False

    # ------------------------------------------------------------------
    def _retract(self) -> None:
        """Re-orthogonalise the weight (QR retraction) – cheap for small ranks."""
        # Only called when geoopt is unavailable.
        with torch.no_grad():
            # QR guarantees orthogonal columns; keep leading `rank` columns.
            q, _ = torch.linalg.qr(self.weight.data, mode="reduced")
            self.weight.data.copy_(q[:, : self.weight.shape[1]])

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if not self._use_geoopt:
            # First forward call prints a warning exactly once.
            if self._warned.item() == 0:
                print(
                    "[StiefelProjector] geoopt not installed – falling back to "
                    "unconstrained parameter with QR re-projection. Install geoopt "
                    "for true manifold optimisation.")
                self._warned.fill_(1)
            self._retract()
        return x @ self.weight  # [B, rank]


class VQLite(nn.Module):
    """Extremely small VQ-VAE encoder, 32× compression, 16-entry codebook."""

    def __init__(self, codebook_size: int = 16, code_dim: int = 32, beta: float = 0.25):
        super().__init__()
        self.code_dim = code_dim
        self.beta = beta

        # ───── Encoder – 32×32 → 8×8 spatial, global-avg pooled ─────
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, code_dim, 1),
        )
        self.register_buffer("codebook", torch.randn(codebook_size, code_dim))

    # ------------------------------------------------------------------
    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # (z_q, idx, vq_loss)
        z_e = self.encoder(x)  # [B, C, 8, 8]
        z_e = z_e.mean(dim=(-1, -2))  # global average -> [B, C]

        # Vector quantisation
        dist = ((z_e.unsqueeze(1) - self.codebook.unsqueeze(0)) ** 2).sum(-1)  # [B, K]
        idx = dist.argmin(dim=-1)
        z_q = self.codebook[idx]  # [B, C]

        # Commitment loss (straight-through)
        loss = (z_q.detach() - z_e).pow(2).mean() + self.beta * (
            z_q - z_e.detach()
        ).pow(2).mean()
        z_q = z_e + (z_q - z_e).detach()
        return z_q, idx, loss

    # ------------------------------------------------------------------
    def decode(self, _):  # decoder never called in CLoVe-Sub (adapter mode)
        raise NotImplementedError("Decoder is not required – replay is latent → feature.")


# -----------------------------------------------------------------------------
#  Replay buffers
# -----------------------------------------------------------------------------

class LatentBuffer:
    """Fixed-capacity buffer that stores latent codes instead of raw images."""

    def __init__(self, max_bytes: int, code_dim: int):
        self.max_bytes = max_bytes
        self.code_dim = code_dim
        self.storage: List[Tuple[torch.Tensor, int]] = []

    # --------------------------------------------------------------
    def _bytes(self) -> int:
        return len(self.storage) * self.code_dim * 4  # float32 on disk / RAM

    # --------------------------------------------------------------
    def add(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        for z, y in zip(codes.cpu(), labels.cpu()):
            while self._bytes() + self.code_dim * 4 > self.max_bytes and self.storage:
                self.storage.pop(0)  # FIFO eviction
            self.storage.append((z.clone(), int(y)))

    # --------------------------------------------------------------
    def sample(self, n: int) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.storage:
            return None, None
        batch = random.sample(self.storage, min(n, len(self.storage)))
        z, y = zip(*batch)
        return torch.stack(list(z)), torch.tensor(list(y))


class ImageBuffer:
    """Classic rehearsal buffer that stores raw images."""

    def __init__(self, max_imgs: int):
        self.max_imgs = max_imgs
        self.storage: List[Tuple[torch.Tensor, int]] = []

    def add(self, imgs: torch.Tensor, labels: torch.Tensor) -> None:
        for img, y in zip(imgs.cpu(), labels.cpu()):
            if len(self.storage) >= self.max_imgs:
                self.storage.pop(0)
            self.storage.append((img.clone(), int(y)))

    def sample(self, n: int) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.storage:
            return None, None
        batch = random.sample(self.storage, min(n, len(self.storage)))
        x, y = zip(*batch)
        return torch.stack(list(x)), torch.tensor(list(y))


# -----------------------------------------------------------------------------
#  Generic continual-learning algorithm base-class
# -----------------------------------------------------------------------------

class _BaseAlgo(nn.Module):
    """Base class: backbone, head, optimiser, generic training loop."""

    def __init__(self, exp_cfg: Dict[str, Any], method_cfg: Dict[str, Any]):
        super().__init__()
        self.exp_cfg = exp_cfg
        self.method_cfg = method_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = tvm.resnet18(weights=None, num_classes=0).to(self.device)
        self.feat_dim = 512
        self.head = nn.Linear(self.feat_dim, exp_cfg["dataset"]["num_classes_total"]).to(
            self.device
        )

        self.loss_ce = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(
            self.parameters(), lr=exp_cfg["optim"]["lr"], momentum=0.9
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        feat = self.backbone(x)
        return self.head(feat)

    # ------------------------------------------------------------------
    def before_task(self, _tid: int) -> None:  # hooks for Fisher, etc.
        self.train()

    def after_task(self, _tid: int) -> None:  # placeholder
        pass

    # ------------------------------------------------------------------
    def train_task(self, tid: int, tr_loader, _val_loader, logger):  # noqa: D401
        epochs = self.exp_cfg["epochs_per_task"]
        for _ in range(epochs):
            for x, y in tr_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                loss = self.loss_ce(self.forward(x), y)
                loss.backward()
                self.opt.step()
            logger.log("train_loss", loss.item())  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, tid: int, test_loader, logger):  # noqa: D401
        self.eval()
        correct = 0
        total = 0
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.forward(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        acc = 100.0 * correct / total
        logger.log(f"acc_task_{tid}", acc)
        self.train()


# -----------------------------------------------------------------------------
#  Concrete algorithms
# -----------------------------------------------------------------------------

class CLoVeSub(_BaseAlgo):
    """Proposed method – compressed latent rehearsal buffer with adapter."""

    def __init__(self, exp_cfg: Dict[str, Any], method_cfg: Dict[str, Any]):
        super().__init__(exp_cfg, method_cfg)
        rank = method_cfg.get("rank", 16)
        self.projector = StiefelProjector(self.feat_dim, rank).to(self.device)
        self.vq = VQLite().to(self.device)
        self.adapter = nn.Sequential(
            nn.Linear(self.vq.code_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, self.feat_dim)
        ).to(self.device)
        self.buffer = LatentBuffer(method_cfg["buffer_bytes"], self.vq.code_dim)
        self.sparsity = method_cfg.get("sparsity", 0.2)

        # Re-initialise optimiser now that projector / adapter are added.
        self.opt = torch.optim.SGD(
            self.parameters(), lr=exp_cfg["optim"]["lr"], momentum=0.9
        )

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor | None = None, z_lat: torch.Tensor | None = None
    ) -> torch.Tensor:  # noqa: D401
        if z_lat is None:
            feat = self.backbone(x)  # type: ignore[arg-type]
        else:
            feat = self.adapter(z_lat.to(self.device))
        feat_p = self.projector(feat)
        return self.head(feat_p)

    # ------------------------------------------------------------------
    def train_task(self, tid: int, tr_loader, _val_loader, logger):  # noqa: D401
        epochs = self.exp_cfg["epochs_per_task"]
        for _ in range(epochs):
            for x, y in tr_loader:
                x, y = x.to(self.device), y.to(self.device)

                # ─── Encode current batch and store latents ───
                z_q, _idx, vq_loss = self.vq.encode(x)
                self.buffer.add(z_q.detach(), y.detach())

                # ─── Sample replay latents ───
                z_r, y_r = self.buffer.sample(len(x))
                logits_replay = None
                if z_r is not None:
                    z_r, y_r = z_r.to(self.device), y_r.to(self.device)
                    logits_replay = self.forward(z_lat=z_r)

                # ─── Forward & loss ───
                logits_cur = self.forward(x=x)
                loss = self.loss_ce(logits_cur, y) + 0.1 * vq_loss
                if logits_replay is not None:
                    loss = loss + self.loss_ce(logits_replay, y_r)

                # ─── Back-prop & sparsity mask ───
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                sparsify_grads(self.backbone, theta=self.sparsity)
                self.opt.step()
            logger.log("train_loss", loss.item())  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
#  Baselines – ER-Ring & SparCL
# -----------------------------------------------------------------------------

class ERRing(_BaseAlgo):
    def __init__(self, exp_cfg: Dict[str, Any], method_cfg: Dict[str, Any]):
        super().__init__(exp_cfg, method_cfg)
        imgs_per_class = method_cfg.get("imgs_per_class", 15)
        max_imgs = imgs_per_class * exp_cfg["dataset"]["num_classes_total"]
        self.buffer = ImageBuffer(max_imgs)

    # ------------------------------------------------------------------
    def train_task(self, tid: int, tr_loader, _val_loader, logger):  # noqa: D401
        epochs = self.exp_cfg["epochs_per_task"]
        for _ in range(epochs):
            for x, y in tr_loader:
                self.buffer.add(x, y)
                x_b, y_b = self.buffer.sample(len(x))
                if x_b is not None:
                    x = torch.cat([x, x_b.to(x.device)])  # reuse current device (CPU / GPU)
                    y = torch.cat([y, y_b.to(y.device)])

                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                loss = self.loss_ce(self.forward(x), y)
                loss.backward()
                self.opt.step()
            logger.log("train_loss", loss.item())  # type: ignore[arg-type]


class SparCL(ERRing):
    """ER-Ring + gradient sparsity mask."""

    def train_task(self, tid: int, tr_loader, _val_loader, logger):  # noqa: D401
        epochs = self.exp_cfg["epochs_per_task"]
        for _ in range(epochs):
            for x, y in tr_loader:
                self.buffer.add(x, y)
                x_b, y_b = self.buffer.sample(len(x))
                if x_b is not None:
                    x = torch.cat([x, x_b.to(x.device)])
                    y = torch.cat([y, y_b.to(y.device)])

                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                loss = self.loss_ce(self.forward(x), y)
                loss.backward()
                sparsify_grads(self.backbone, theta=0.2)
                self.opt.step()
            logger.log("train_loss", loss.item())  # type: ignore[arg-type]