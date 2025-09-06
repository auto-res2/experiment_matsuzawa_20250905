"""
train.py – models, memory modules and training utilities
This file contains everything that is required for constructing the
networks that are used in the experiments as well as the helper
functions that actually perform the training on a continual data
stream.
Only standard-library modules plus the libraries declared in
pyproject.toml are imported. All CUDA operations are guarded such that
the script can still be executed on CPU-only machines (it will simply
exit early, see main.py).
"""
from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Back-bones & classifier
# ---------------------------------------------------------------------------

try:
    import torchvision
except ImportError as _e:  # pragma: no cover
    raise RuntimeError("torchvision is required – please install it before running the experiments") from _e


class ExpandingClassifier(nn.Module):
    """A linear classifier whose output dimension can grow on-the-fly."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 0
        # start with a single dummy output that will be replaced once grow() is called
        self.fc = nn.Linear(in_dim, 1)

    # ---------------------------------------------------------------------
    # growing logic
    # ---------------------------------------------------------------------

    def _grow(self, n_new: int) -> None:
        old_w, old_b = self.fc.weight.data.clone(), self.fc.bias.data.clone()
        new = nn.Linear(self.in_dim, self.out_dim + n_new).to(self.fc.weight.device)
        if self.out_dim:
            new.weight.data[: self.out_dim] = old_w
            new.bias.data[: self.out_dim] = old_b
        self.fc = new
        self.out_dim += n_new

    def ensure_capacity(self, n_classes: int) -> None:
        """Make sure the classifier can output *n_classes* logits."""
        if n_classes > self.out_dim:
            self._grow(n_classes - self.out_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – short description is fine
        if self.out_dim == 0:
            raise RuntimeError("Classifier has 0 outputs – ensure_capacity() must be called before the first forward")
        return self.fc(x)


# ---------------------------------------------------------------------------
#  Backbone construction helpers
# ---------------------------------------------------------------------------

def build_backbone(name: str = "resnet18") -> nn.Module:
    """Load an ImageNet-pre-trained backbone and attach a 256-dim projection."""

    name = name.lower()
    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=weights)
        net.fc = nn.Identity()
        proj = nn.Linear(512, 256)
        backbone = nn.Sequential(net, proj, nn.LayerNorm(256, elementwise_affine=False))
    elif name in {"mobilenet", "mobilenet_v2", "mobilenet-v2"}:
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        net = torchvision.models.mobilenet_v2(weights=weights)
        net.classifier = nn.Identity()
        proj = nn.Linear(net.last_channel, 256)
        backbone = nn.Sequential(net, proj, nn.LayerNorm(256, elementwise_affine=False))
    else:
        raise ValueError(f"Unknown backbone '{name}' – supported: resnet18, mobilenet_v2")

    return backbone


# ---------------------------------------------------------------------------
#  LOSR – latent orthogonal sketch replay memory
# ---------------------------------------------------------------------------


def _quant8(fp: torch.Tensor):
    s = fp.abs().max().clamp(min=1e-8)
    q = ((fp / s) * 127).round() + 128  # scale to [1;255]
    return q.to(torch.uint8), s.item()


def _unpack4bit(packed: torch.Tensor, length: int) -> torch.Tensor:
    """Inverse of the 4-bit packer. Needed at inference time only."""
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    out = torch.empty(length, dtype=torch.uint8)
    out[0::2] = high
    out[1::2] = low
    return out


class AnchorBank(nn.Module):
    """Analytic per-class memory that only stores tiny statistics (kB range)."""

    def __init__(self, feat_dim: int = 256, r: int = 2, budget_kb: int = 32):
        super().__init__()
        self.full = feat_dim
        self.comp = 32  # compressed dimensionality
        self.r = r
        self.budget = budget_kb * 1024

        P = torch.randn(feat_dim, self.comp)
        # orthogonal basis for compression
        P, _ = torch.linalg.qr(P)
        self.register_buffer("P", P)

        # in-memory structures
        self.mu_q: Dict[int, torch.Tensor] = {}
        self.scale: Dict[int, float] = {}
        self.U_idx: Dict[int, torch.Tensor] = {}
        # learned 4-bit code-book (16 entries)
        self.codebook = nn.Parameter(torch.randn(16))

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def dequant_mu(self, cls: int) -> torch.Tensor:
        return (self.mu_q[cls].float() - 128) * (self.scale[cls] / 127.0)

    def bytes(self) -> int:
        b = 0
        for cls in self.mu_q:
            b += self.comp  # µ (uint8)
            b += (self.r * self.comp) // 2  # 4-bit compressed U
            b += 4  # scale (fp32)
        return b

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """Update EMA anchors and keep the size budget in check."""
        comp = F.normalize(feats @ self.P, dim=1)
        for z, y in zip(comp, labels):
            c = int(y)
            if c not in self.mu_q:  # new class
                mu_q, s = _quant8(z.cpu())
                self.mu_q[c] = mu_q
                self.scale[c] = s
                self.U_idx[c] = torch.zeros(self.r * self.comp // 2, dtype=torch.uint8)
            else:  # EMA update
                mu_fp = 0.99 * self.dequant_mu(c) + 0.01 * z.cpu()
                mu_q, s = _quant8(mu_fp)
                self.mu_q[c] = mu_q
                self.scale[c] = s

        if self.bytes() > self.budget:
            raise MemoryError("AnchorBank exceeds the set memory budget – aborting run as specified")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, cls: int, n: int = 16) -> torch.Tensor:
        """Generate *n* synthetic compressed features and uncompress them."""
        mu = self.dequant_mu(cls)
        U_code = _unpack4bit(self.U_idx[cls], self.r * self.comp).view(self.r, self.comp).long()
        U = self.codebook[U_code]
        eps = torch.randn(n, self.r)
        z_comp = mu + eps @ U
        z_full = z_comp @ self.P.T
        return F.normalize(z_full, dim=1)


class FeatureSynthesiser(nn.Module):
    """Tiny hyper-network that creates full size latent features conditioned on a class id."""

    def __init__(self, feat_dim: int = 256, noise_dim: int = 8, n_cls: int = 1_000):
        super().__init__()
        self.embed = nn.Embedding(n_cls, 16)
        self.fc1 = nn.Linear(noise_dim + 16, 128)
        self.fc2 = nn.Linear(128, feat_dim)

    def forward(self, eps: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:  # noqa: D401
        h = torch.cat([eps, self.embed(cls)], dim=1)
        h = F.relu(self.fc1(h))
        return F.normalize(self.fc2(h), dim=1)


class LOSRMemory(nn.Module):
    """Full LOSR memory consisting of anchors + feature synthesiser."""

    def __init__(self, feat_dim: int = 256, r: int = 2, budget_kb: int = 32):
        super().__init__()
        self.bank = AnchorBank(feat_dim, r, budget_kb)
        self.synth = FeatureSynthesiser(feat_dim)

    # proxy helpers -------------------------------------------------------
    def bytes(self) -> int:  # -> kB in caller
        return self.bank.bytes()

    @torch.no_grad()
    def generate(self, n_per_cls: int = 16):
        if not self.bank.mu_q:
            raise ValueError("generate() called before any anchor exists")
        feats, labels = [], []
        for c in self.bank.mu_q:
            feats.append(self.bank.sample(c, n_per_cls))
            labels.extend([c] * n_per_cls)
        return torch.cat(feats), torch.tensor(labels)

    # pass-through update --------------------------------------------------
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        self.bank.update(feats.cpu(), labels.cpu())


# ---------------------------------------------------------------------------
#  Simple reservoir buffer for ER baseline
# ---------------------------------------------------------------------------

class ERBuffer:  # experience replay
    def __init__(self, byte_budget: int, img_shape: Tuple[int, int, int] = (3, 32, 32)):
        self.byte_budget = byte_budget
        self.img_bytes = math.prod(img_shape) * 2  # fp16 storage cost
        self.max_items = max(1, byte_budget // self.img_bytes)
        self.storage: List[Tuple[torch.Tensor, int]] = []
        self.n_seen = 0

    # ------------------------------------------------------------------
    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for img, lbl in zip(x, y):
            self.n_seen += 1
            if len(self.storage) < self.max_items:
                self.storage.append((img.half().cpu(), int(lbl)))
            else:
                j = random.randrange(self.n_seen)
                if j < self.max_items:
                    self.storage[j] = (img.half().cpu(), int(lbl))

    # ------------------------------------------------------------------
    def sample(self, k: int):
        if k > len(self.storage):
            k = len(self.storage)
        idx = random.sample(range(len(self.storage)), k)
        imgs, lbls = zip(*(self.storage[i] for i in idx))
        return torch.stack(list(imgs)).float(), torch.tensor(lbls)

    # ------------------------------------------------------------------
    def bytes(self) -> int:
        return self.max_items * self.img_bytes


# ---------------------------------------------------------------------------
#  Training helpers – AMP, FLOPs profiler & main loop
# ---------------------------------------------------------------------------

try:
    from ptflops import get_model_complexity_info

    def _profile_flops_model(backbone, input_size):
        macs, _ = get_model_complexity_info(
            backbone, input_size, as_strings=False, print_per_layer_stat=False
        )
        return macs * 2  # FLOPs ≈ 2 × MACs

except Exception:  # pragma: no cover – ptflops might be unavailable

    def _profile_flops_model(backbone, input_size):  # type: ignore[override]
        print("[Warning] ptflops not found – setting FLOPs to zero for this run")
        return 0


def profile_flops(backbone, input_size=(3, 32, 32)) -> float:
    """Return giga-FLOPs for a single forward pass of *backbone*."""
    return _profile_flops_model(backbone, input_size) / 1e9


# ---------------------------------------------------------------------------
#  Main training loop for one task stream
# ---------------------------------------------------------------------------
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def train_stream(
    backbone: nn.Module,
    classifier: ExpandingClassifier,
    optimiser: torch.optim.Optimizer,
    loader: DataLoader,
    strategy: str,
    losr: LOSRMemory | None,
    buffer: ERBuffer | None,
    device: str,
    scaler: GradScaler,
    print_every: int = 100,
) -> None:
    """Single pass through *loader* using the chosen replay *strategy*."""

    backbone.train()
    classifier.train()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        classifier.ensure_capacity(int(y.max()) + 1)

        with autocast():
            feats = backbone(x)
            loss = F.cross_entropy(classifier(feats), y)

            # ----------------------------------------------------------------
            #  Optional replay / synthetic generation
            # ----------------------------------------------------------------
            if strategy == "LOSR" and losr is not None and losr.bank.mu_q:
                syn_f, syn_y = losr.generate()
                syn_f, syn_y = syn_f.to(device), syn_y.to(device)
                classifier.ensure_capacity(int(syn_y.max()) + 1)
                loss = loss + F.cross_entropy(classifier(syn_f), syn_y)

                eps = torch.randn(len(syn_y), 8, device=device)
                gen_f = losr.synth(eps, syn_y)
                loss = loss + F.cross_entropy(classifier(gen_f), syn_y)

            elif strategy == "ER" and buffer is not None and len(buffer.storage) >= len(y):
                bx, by = buffer.sample(len(y))
                bx, by = bx.to(device), by.to(device)
                classifier.ensure_capacity(int(by.max()) + 1)
                loss = loss + F.cross_entropy(classifier(backbone(bx)), by)

        # --------------------------------------------------------------------
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        # ---------------- memory update -------------------------------------
        if strategy == "LOSR" and losr is not None:
            losr.update(feats.detach(), y.detach())

        if strategy == "ER" and buffer is not None:
            buffer.add(x.cpu(), y.cpu())

        # ---------------- diagnostics ---------------------------------------
        if (i + 1) % print_every == 0:
            anchors = len(losr.bank.mu_q) if losr else 0
            buf_items = len(buffer.storage) if buffer else 0
            print(
                f"iter={i+1:4d}  cls_out={classifier.out_dim:3d}  anchors={anchors:3d}  buffer={buf_items:4d}  loss={loss.item():.4f}"
            )
