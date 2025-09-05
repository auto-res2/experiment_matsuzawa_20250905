"""src/train.py
All model components and the training logic live here.
Every public function explicitly receives the `config` dictionary that
is read once by `src.main` so that no file needs to import PyYAML on its
own.  This design prevents circular-imports and makes unit-testing easy
because the whole behaviour can be controlled via the passed‐in config.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

# 3rd-party (optional) -------------------------------------------------
try:
    import geoopt  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("Package 'geoopt' is required – pip install geoopt>=0.5,<0.7") from exc

# ---------------------------------------------------------------------
#  DEVICE MANAGEMENT – central place in this module so that every other
#  sub-routine can simply call `to(DEVICE)`.  The caller (src.main)
#  should set the RNG seeds immediately after importing torch so that
#  CUDA initialisation is deterministic.
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================================================
#  1.  MODEL BUILDING BLOCKS
# ===============================================================


class VQLayer(nn.Module):
    """Vector-Quantisation layer (straight-through estimator)."""

    def __init__(self, n_codes: int = 16, code_dim: int = 32):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_codes, code_dim))
        nn.init.uniform_(self.codebook, -1 / n_codes, 1 / n_codes)

    def forward(
        self, z_e: torch.Tensor, tau: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: (B, C, H, W) – flatten spatial dims
        B, C, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)
        logits = torch.matmul(z, self.codebook.t())  # (N, n_codes)
        hard = F.gumbel_softmax(logits, tau=tau, hard=True)
        z_q = torch.matmul(hard, self.codebook)  # (N, C)
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # commitment loss
        loss = F.mse_loss(z_q.detach(), z_e) + F.mse_loss(z_q, z_e.detach())
        codes = hard.argmax(dim=1).view(B, H, W).to(torch.uint8)
        return z_q + (z_q - z_e).detach(), codes, loss


class Encoder(nn.Module):
    def __init__(self, code_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, code_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OrthoProjector(nn.Module):
    """Low-rank projector constrained to the Stiefel manifold."""

    def __init__(self, in_dim: int, rank: int = 16):
        super().__init__()
        self.P = geoopt.ManifoldParameter(torch.empty(in_dim, rank), manifold=geoopt.Stiefel())
        nn.init.orthogonal_(self.P.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.P  # (B, rank)


class LatentAdapter(nn.Module):
    def __init__(self, code_dim: int = 32, feat_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(code_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )

    def forward(self, codes: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # codes: (B, H, W) uint8  – average embed per spatial position
        embed = F.embedding(codes.long(), codebook)  # (B, H, W, C)
        vec = embed.mean(dim=(1, 2))  # (B, C)
        return self.mlp(vec)


# ------------------------------------------------------------------
#  Sparsity helper
# ------------------------------------------------------------------

def sparsify_grads(model: nn.Module, theta: float = 0.2) -> None:
    """Top-k magnitude gradient masking (keeps (1-theta) fraction)."""

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            k = int((1.0 - theta) * p.grad.numel())
            if k == 0:
                p.grad.zero_()
                continue
            thresh = torch.topk(p.grad.abs().flatten(), k, largest=True).values.min()
            mask = p.grad.abs() >= thresh
            p.grad.mul_(mask)


# ===============================================================
#  2.  REHEARSAL BUFFER (latent codes)
# ===============================================================


class LatentBuffer:
    """Fixed-budget buffer that stores *latent* VQ codes instead of pixels."""

    def __init__(self, K: int, num_classes: int, code_shape: Tuple[int, int]):
        self.K = K
        self.num_classes = num_classes
        self.H, self.W = code_shape
        self.storage: Dict[int, List[torch.Tensor]] = {c: [] for c in range(num_classes)}
        self.bytes_per_code = self.H * self.W  # uint8 – 1 byte each

    # --------------------------------------------------------------
    def add(self, class_id: int, codes: torch.Tensor) -> None:
        bucket = self.storage[class_id]
        if len(bucket) < self.K:
            bucket.append(codes.cpu())
        else:  # ring overwrite
            idx = random.randint(0, self.K - 1)
            bucket[idx] = codes.cpu()

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        choices, labels = [], []
        for _ in range(n):
            cls = random.randint(0, self.num_classes - 1)
            if len(self.storage[cls]) == 0:
                continue
            code = random.choice(self.storage[cls])
            choices.append(code)
            labels.append(cls)
        if not choices:
            return (
                torch.empty(0, self.H, self.W, dtype=torch.uint8),
                torch.empty(0, dtype=torch.long),
            )
        return torch.stack(choices, 0), torch.tensor(labels, dtype=torch.long)

    # --------------------------------------------------------------
    def size_bytes(self) -> int:
        return sum(len(v) for v in self.storage.values()) * self.bytes_per_code


# ===============================================================
#  3.  MAIN MODEL – CLoVe-Sub
# ===============================================================


class CLoVeSub(nn.Module):
    """Full CLoVe-Sub continual-learning architecture."""

    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config  # keep reference
        cfg_m = config["models"]

        # ------------------------------------------------------------------
        #  Backbone (ResNet-18)
        # ------------------------------------------------------------------
        self.backbone = torchvision.models.resnet18(pretrained=cfg_m.get("pretrained", False))
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Low-rank projector on the Stiefel manifold ------------------------
        self.projector = OrthoProjector(feat_dim, cfg_m["projector_rank"])
        self.head = nn.Linear(cfg_m["projector_rank"], 100)  # fixed for CIFAR-100

        # VQ-VAE components --------------------------------------------------
        self.encoder = Encoder(cfg_m["vq"]["code_len"])
        self.vq = VQLayer(cfg_m["vq"]["n_codes"], cfg_m["vq"]["code_len"])
        self.adapter = LatentAdapter(cfg_m["vq"]["code_len"], feat_dim)

        # Buffer -------------------------------------------------------------
        code_H = config["dataset"]["img_size"] // 4  # encoder downsamples by 4
        self.buffer = LatentBuffer(cfg_m["buffer"]["K"], 100, (code_H, code_H))

        # Optimisers ---------------------------------------------------------
        p_main = list(self.projector.parameters()) + list(self.head.parameters())
        self.opt_main = torch.optim.SGD(
            p_main,
            lr=config["optim"]["lr_backbone"],
            momentum=config["optim"]["momentum"],
            weight_decay=config["optim"]["weight_decay"],
        )
        self.opt_vq = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.vq.parameters())
            + list(self.adapter.parameters()),
            lr=config["optim"]["lr_vq"],
        )

        self.amp = config["global"].get("amp", True) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.to(DEVICE)

    # ------------------------------------------------------------------
    #  Forward helpers
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pixels → logits
        feats = self.backbone(x)
        feats = self.projector(feats)
        return self.head(feats)

    def _feature_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            codebook = self.vq.codebook.detach()
        return self.adapter(codes.to(DEVICE), codebook)

    # ------------------------------------------------------------------
    #  Full training of a single task
    # ------------------------------------------------------------------
    def train_task(
        self, loader_cur: DataLoader, loader_val: DataLoader, task_id: int, logger,  # noqa: ANN001
    ) -> None:
        cfg_g = self.cfg["global"]
        cfg_m = self.cfg["models"]
        epochs = cfg_g["epochs_per_task"]
        tau = cfg_m["vq"]["tau"]
        beta = cfg_m["vq"]["beta"]
        sparsity = cfg_m["sparsity"]

        for epoch in range(epochs):
            self.train()
            for xb, yb in loader_cur:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                bsz = xb.size(0)

                # --------------- forward + losses ------------------------
                if self.amp:
                    with torch.cuda.amp.autocast():
                        z_e = self.encoder(xb.float())
                        z_q, codes, vq_loss = self.vq(z_e, tau)
                        logits_cur = self(xb.float())
                        loss_cls = F.cross_entropy(logits_cur, yb)
                        feats_real = self.backbone(xb.float())
                        feats_pred = self.adapter(codes, self.vq.codebook)
                        loss_adapter = F.mse_loss(feats_pred, feats_real.detach())
                        loss = loss_cls + beta * vq_loss + 0.1 * loss_adapter
                else:
                    # identical computation without autocast
                    z_e = self.encoder(xb.float())
                    z_q, codes, vq_loss = self.vq(z_e, tau)
                    logits_cur = self(xb.float())
                    loss_cls = F.cross_entropy(logits_cur, yb)
                    feats_real = self.backbone(xb.float())
                    feats_pred = self.adapter(codes, self.vq.codebook)
                    loss_adapter = F.mse_loss(feats_pred, feats_real.detach())
                    loss = loss_cls + beta * vq_loss + 0.1 * loss_adapter

                # ---------------- backward / optimisation ---------------
                self.scaler.scale(loss).backward()
                sparsify_grads(self.backbone, sparsity)
                self.scaler.step(self.opt_main)
                self.scaler.step(self.opt_vq)
                self.scaler.update()
                self.opt_main.zero_grad(set_to_none=True)
                self.opt_vq.zero_grad(set_to_none=True)

                # ---------------- buffer update -------------------------
                for i in range(bsz):
                    self.buffer.add(int(yb[i]), codes[i].cpu())

            # ------------- quick validation (single batch) --------------
            self.eval()
            xb_val, yb_val = next(iter(loader_val))
            xb_val, yb_val = xb_val.to(DEVICE), yb_val.to(DEVICE)
            with torch.no_grad():
                logits = self(xb_val.float())
                acc = (logits.argmax(1) == yb_val).float().mean().item()
            logger.log(f"task{task_id}/val_acc", acc)

    # ------------------------------------------------------------------
    #  Replay – returns projected features + labels
    # ------------------------------------------------------------------
    def replay_step(self, n: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        codes, labels = self.buffer.sample(n)
        if codes.numel() == 0:
            return (
                torch.empty(0, self.cfg["models"]["projector_rank"]).to(DEVICE),
                torch.empty(0, dtype=torch.long).to(DEVICE),
            )
        feats = self._feature_from_codes(codes)
        feats = self.projector(feats)
        return feats, labels.to(DEVICE)

    # ------------------------------------------------------------------
    #  Evaluation on a single task
    # ------------------------------------------------------------------
    def evaluate_task(self, loader_test: DataLoader) -> float:
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader_test:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = self(xb.float()).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        return correct / total
