"""
src/train.py – model definitions, training utilities and baseline stubs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Deque, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

# -----------------------------------------------------------------------------
#  Utility: sparse–gradient masking (SparCL-style)
# -----------------------------------------------------------------------------

def mask_gradients(backbone: nn.Module, theta: float = 0.2):
    """Prune (1-θ) of gradients whose cosine similarity is below median."""
    with torch.no_grad():
        grads = [p.grad.view(-1) for p in backbone.parameters() if p.grad is not None]
        if not grads:
            return
        g = torch.cat(grads)
        norm = g.norm() + 1e-6
        cos = g / norm
        k = int(theta * cos.numel())
        thresh = torch.topk(torch.abs(cos), k, largest=False).values.max()
        for p in backbone.parameters():
            if p.grad is not None:
                mask = torch.abs(p.grad) < thresh
                p.grad[mask] = 0.0

# -----------------------------------------------------------------------------
#  Latent replay buffer (fixed byte capacity)
# -----------------------------------------------------------------------------
class LatentReplayBuffer:
    def __init__(self, capacity_bytes: int, code_dim: int):
        self.capacity = max(1, capacity_bytes // (code_dim * 4))  # float32 = 4 B
        self.store: Deque[Tuple[torch.Tensor, torch.Tensor]] = torch.deque(maxlen=self.capacity)  # type: ignore[arg-type]

    def __len__(self):
        return len(self.store)

    @torch.no_grad()
    def add(self, z: torch.Tensor, y: torch.Tensor):
        for zi, yi in zip(z, y):
            self.store.append((zi.cpu(), yi.cpu()))

    def sample(self, n: int):
        idx = torch.randint(0, len(self.store), (n,))
        z, y = zip(*[self.store[i] for i in idx])
        return torch.stack(z).cuda(non_blocking=True), torch.tensor(y).cuda(non_blocking=True)

# -----------------------------------------------------------------------------
#  VQ-VAE-Lite building blocks
# -----------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 16, code_dim: int = 32, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim))
        self.beta = beta

    def forward(self, z_e: torch.Tensor):
        d = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        ind = d.argmin(1)
        z_q = self.codebook[ind]
        loss = self.beta * F.mse_loss(z_e.detach(), z_q) + F.mse_loss(z_e, z_q.detach())
        z_q = z_e + (z_q - z_e).detach()  # straight-through
        return z_q, ind, loss


class Encoder(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 64, code_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, code_dim, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, code_dim: int = 32, hidden: int = 64, out_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (code_dim, 1, 1)),
            nn.ConvTranspose2d(code_dim, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(nn.Module):
    def __init__(self, codebook_size: int = 16, code_dim: int = 32):
        super().__init__()
        self.enc = Encoder(code_dim=code_dim)
        self.quant = VectorQuantizer(codebook_size, code_dim)
        self.dec = Decoder(code_dim=code_dim)

    def forward(self, x):
        z_e = self.enc(x)
        z_q, ind, vq_loss = self.quant(z_e)
        x_hat = self.dec(z_q)
        recon_loss = F.mse_loss(x_hat, x)
        return {"z_q": z_q, "indices": ind, "loss": recon_loss + vq_loss}

# -----------------------------------------------------------------------------
#  Orthogonal projector & latent adapter
# -----------------------------------------------------------------------------
try:
    import geoopt
except ImportError:
    geoopt = None  # graceful degradation – will raise later if actually used


class OrthogonalProjector(nn.Module):
    """Low-rank projector P∈Stiefel(d,r)."""

    def __init__(self, feat_dim: int, rank: int):
        if geoopt is None:
            raise RuntimeError("geoopt is required for OrthogonalProjector. Install geoopt >=0.6.")
        super().__init__()
        manifold = geoopt.Stiefel()
        init = torch.empty(feat_dim, rank).orthogonal_()
        self.P = geoopt.ManifoldParameter(init, manifold=manifold)

    def forward(self, x):
        return x @ self.P


class LatentAdapter(nn.Module):
    def __init__(self, code_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
        )

    def forward(self, z):
        return self.net(z)

# -----------------------------------------------------------------------------
#  The complete CLoVe-Sub model (continual-learning API)
# -----------------------------------------------------------------------------
class CLoVeSub(nn.Module):
    """Compressed Latent Orthogonal-Subspace replay model."""

    def __init__(self, exp_cfg: dict, method_cfg: dict):
        super().__init__()
        self.exp_cfg = exp_cfg
        self.method_cfg = method_cfg

        # Backbone
        self.backbone = tv.resnet18(weights=None)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Task-specific projectors
        self.projectors: List[OrthogonalProjector] = []
        self.classifier = nn.Linear(method_cfg["rank"], exp_cfg["dataset"]["num_classes_total"], bias=False)

        # VQ-VAE-Lite & adapter
        self.vqvae = VQVAE(codebook_size=16, code_dim=32)
        self.adapter = LatentAdapter(32, feat_dim)

        # Fixed-size replay buffer
        self.buffer = LatentReplayBuffer(capacity_bytes=method_cfg["buffer_bytes"], code_dim=32)

        # Optimisers
        self.opt_main = torch.optim.SGD(
            list(self.backbone.parameters())
            + list(self.classifier.parameters())
            + list(self.adapter.parameters()),
            lr=exp_cfg["optim"]["lr"],
            momentum=0.9,
            weight_decay=5e-4,
        )
        self.opt_vq = torch.optim.Adam(self.vqvae.parameters(), lr=1e-4)

    # =============================================================
    #  Continual-learning public API – called by src/main
    # =============================================================
    def before_task(self, task_id: int):
        rank = self.method_cfg["rank"]
        proj = OrthogonalProjector(self.backbone.backbone.out_features if hasattr(self.backbone, 'backbone') else 512, rank).cuda()  # noqa: E501
        self.projectors.append(proj)
        self.opt_main.add_param_group({"params": proj.parameters()})

    def forward(self, x, task_id: int):
        feat = self.backbone(x)
        feat = self.projectors[task_id](feat)
        return self.classifier(feat)

    # -------------------------------------------------------------
    def train_task(self, task_id: int, train_ds, val_ds, recorder: "MetricRecorder"):
        loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.exp_cfg["batch_size"],
            shuffle=True,
            num_workers=4,
        )
        self.train()

        for epoch in range(self.exp_cfg["epochs_per_task"]):
            for img, label in loader:
                img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)

                # --------- forward passes ---------
                logits = self(img, task_id)
                loss_cls = F.cross_entropy(logits, label)

                vq_out = self.vqvae(img)
                loss_vq = vq_out["loss"]

                feat_real = self.backbone(img).detach()
                feat_hat = self.adapter(vq_out["z_q"])
                loss_adapter = F.mse_loss(feat_hat, feat_real)

                # Replay
                if len(self.buffer) > 0:
                    z_rep, y_rep = self.buffer.sample(self.exp_cfg["batch_size"] // 2)
                    feat_rep = self.adapter(z_rep)
                    logits_rep = self.classifier(self.projectors[task_id](feat_rep))
                    loss_rep = F.cross_entropy(logits_rep, y_rep)
                else:
                    loss_rep = torch.tensor(0.0, device=img.device)

                loss = (
                    loss_cls
                    + self.method_cfg["lambda_vq"] * loss_vq
                    + self.method_cfg["lambda_adapter"] * loss_adapter
                    + loss_rep
                )

                # --------- optimisation ---------
                self.opt_main.zero_grad(set_to_none=True)
                self.opt_vq.zero_grad(set_to_none=True)
                loss.backward()
                mask_gradients(self.backbone, theta=self.method_cfg["sparsity"])
                self.opt_main.step()
                self.opt_vq.step()

                recorder.step("train_loss", float(loss_cls.item()))
                self.buffer.add(vq_out["z_q"].detach(), label.detach())

    def after_task(self, task_id: int):
        for p in self.projectors[task_id].parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def evaluate(self, task_id: int, test_ds, recorder: "MetricRecorder"):
        self.eval()
        ld = torch.utils.data.DataLoader(test_ds, batch_size=256, num_workers=4)
        total = correct = 0
        for x, y in ld:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            pred = self(x, task_id).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        acc = correct / total
        recorder.step("accuracy", acc, aggregate="last")
        print(f"[CLoVe-Sub] Task {task_id:02d}  Accuracy: {acc:.2%}")

# -----------------------------------------------------------------------------
#  Baseline stubs (to keep factory imports working). Implementations can be
#  swapped-in later without touching main.py.
# -----------------------------------------------------------------------------
class _BaseStub(nn.Module):
    def __init__(self, *_, **__):
        super().__init__()
        print("[WARN] Baseline stub used – implementation missing.")

    def before_task(self, *_):
        pass

    def train_task(self, *_):
        pass

    def after_task(self, *_):
        pass

    def evaluate(self, *_):
        pass


class ERRing(_BaseStub):
    pass


class EWC(_BaseStub):
    pass


class SparCL(_BaseStub):
    pass


class TinyAQM(_BaseStub):
    pass
