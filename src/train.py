"""src/train.py – models, replay memories and training loop"""
from __future__ import annotations

# std -----------------------------------------------------------------------
import math
import random
from typing import Optional, List, Dict, Tuple

# third-party ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

# ============================================================================
# Models --------------------------------------------------------------------


class ExpandingClassifier(nn.Module):
    """A linear classifier that can grow output units on the fly."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = 0
        # start with 1 output to avoid a zero-sized weight matrix on some back-ends
        self.fc = nn.Linear(in_dim, 1, bias=True)
        self.out_dim = 1  # will be replaced before the first forward call

    # ------------------------------------------------------------------ utils
    def _grow(self, n_new: int) -> None:
        """Grow the classifier by *n_new* units, copying existing weights."""
        old_w = self.fc.weight.data.clone()
        old_b = self.fc.bias.data.clone()
        new = nn.Linear(self.in_dim, self.out_dim + n_new, bias=True)
        with torch.no_grad():
            new.weight[: self.out_dim].copy_(old_w)
            new.bias[: self.out_dim].copy_(old_b)
        self.fc = new.to(self.fc.weight.device)
        self.out_dim += n_new

    def ensure_capacity(self, n_classes: int) -> None:
        if n_classes > self.out_dim:
            self._grow(n_classes - self.out_dim)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.out_dim == 1:
            raise RuntimeError(
                "Classifier has not been grown yet – call ensure_capacity() first"
            )
        return self.fc(x)


# ----------------------------------------------------------------------------
# Backbone builders ----------------------------------------------------------


def build_backbone(name: str) -> nn.Module:
    """Return feature extractor with 256-d output."""
    if name.lower() == "resnet18":
        net = torchvision.models.resnet18(weights=None)
        net.fc = nn.Identity()
        proj = nn.Linear(512, 256)
        head = nn.Sequential(net, proj, nn.LayerNorm(256, elementwise_affine=False))
    elif name.lower() == "mobilenet_v2":
        net = torchvision.models.mobilenet_v2(weights=None)
        net.classifier = nn.Identity()
        proj = nn.Linear(net.last_channel, 256)
        head = nn.Sequential(net, proj, nn.LayerNorm(256, elementwise_affine=False))
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return head

# ============================================================================
# LOSR components ------------------------------------------------------------

# Helper for packing / unpacking uint4 tensors -------------------------------

def _pack4bit(t: torch.Tensor) -> torch.Tensor:
    high = (t[0::2] & 0x0F) << 4
    low = t[1::2] & 0x0F
    return high | low


def _unpack4bit(t: torch.Tensor, length: int) -> torch.Tensor:
    high = (t >> 4) & 0x0F
    low = t & 0x0F
    out = torch.empty(length, dtype=torch.uint8, device=t.device)
    out[0::2] = high
    out[1::2] = low
    return out


class AnchorBank(nn.Module):
    """Quantised anchor bank that obeys a strict global byte budget."""

    DEF_DIM_STORE = 32  # compressed latent dimensionality

    def __init__(self, feat_dim_full: int, r: int = 2, budget_kb: int = 32):
        super().__init__()
        self.full = feat_dim_full
        self.comp = self.DEF_DIM_STORE
        self.r = r
        self.budget = budget_kb * 1024  # convert to bytes

        # fixed random orthogonal projection (does not count towards memory)
        P = torch.randn(feat_dim_full, self.comp)
        P, _ = torch.linalg.qr(P)
        self.register_buffer("P", P)  # type: ignore[attr-defined]

        # per-class state (stored on CPU to save GPU RAM)
        self.mu_q: Dict[int, torch.Tensor] = {}
        self.scale: Dict[int, float] = {}
        self.U_idx: Dict[int, torch.Tensor] = {}

        # 16-level scalar code-book. Using a 1-D codebook avoids the erroneous
        # broadcast that caused the matmul dimension mismatch at runtime.
        self.codebook = nn.Parameter(torch.randn(16))

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _quant8(v: torch.Tensor) -> Tuple[torch.Tensor, float]:
        s = v.abs().max().clamp_min(1e-8)
        q = ((v / s) * 127).round() + 128  # 0-255 uint8
        return q.to(torch.uint8), float(s.item())

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """Update anchors with *feats* and *labels* (both tensors on same device)."""
        comp = F.normalize(feats @ self.P, dim=1)  # project to 32-D
        for z, y in zip(comp, labels):
            cls = int(y)
            if cls not in self.mu_q:  # first time we see the class
                mu_q, s = self._quant8(z)
                self.mu_q[cls] = mu_q.cpu()
                self.scale[cls] = s
                self.U_idx[cls] = torch.zeros(
                    self.r * self.comp // 2, dtype=torch.uint8
                )
            else:
                # exponential moving average in fp32
                mu_fp = 0.99 * self.dequant_mu(cls) + 0.01 * z.cpu()
                mu_q, s = self._quant8(mu_fp)
                self.mu_q[cls] = mu_q
                self.scale[cls] = s

        if self.bytes() > self.budget:
            raise MemoryError("AnchorBank exceeds byte budget – aborting")

    # -------------------------------------------------------------------- io
    def dequant_mu(self, cls: int) -> torch.Tensor:
        return (self.mu_q[cls].float() - 128.0) * (self.scale[cls] / 127.0)

    def bytes(self) -> int:
        b = 0
        for cls in self.mu_q:
            b += self.comp  # mu_q uint8
            b += self.r * self.comp // 2  # U_idx packed 4-bit
            b += 4  # scale float32
        return b

    # ---------------------------------------------------------------- sampling
    def sample(self, cls: int, n: int = 32) -> torch.Tensor:
        """Return *n* normalised features for *cls* (on CPU)."""
        mu = self.dequant_mu(cls)  # [comp]
        # --- reconstruct U ∈ ℝ^{r×comp} from the packed 4-bit indices
        U_code = _unpack4bit(self.U_idx[cls], self.r * self.comp)
        U_code = U_code.view(self.r, self.comp).long()
        U = self.codebook[U_code]  # [r, comp]

        eps = torch.randn(n, self.r)
        z_comp = mu + eps @ U  # [n, comp]
        z_full = z_comp @ self.P.T  # back-projection to 256-D
        return F.normalize(z_full, dim=1)


class FeatureSynthesiser(nn.Module):
    """Tiny hyper-network that turns noise + class-id into a 256-D feature."""

    def __init__(self, feat_dim: int, n_cls_max: int = 500, noise: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_cls_max, 16)
        self.fc1 = nn.Linear(noise + 16, 128)
        self.fc2 = nn.Linear(128, feat_dim)

    def forward(self, eps: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = torch.cat([eps, self.embed(cls)], dim=1)
        h = F.relu(self.fc1(h))
        return F.normalize(self.fc2(h), dim=1)


class LOSRMemory(nn.Module):
    """Full memory module combining the analytic anchor bank and synthesiser."""

    def __init__(self, feat_dim_full: int = 256, r: int = 2, budget_kb: int = 32):
        super().__init__()
        self.bank = AnchorBank(feat_dim_full, r, budget_kb)
        self.synth = FeatureSynthesiser(feat_dim_full)

    # ---------------------------------------------------------------- public
    @torch.no_grad()
    def generate(self, n_per_cls: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.bank.mu_q:
            raise ValueError("generate() called before any anchor exists")
        feats: List[torch.Tensor] = []
        labels: List[int] = []
        for cls in self.bank.mu_q:
            feats.append(self.bank.sample(cls, n_per_cls))
            labels.extend([cls] * n_per_cls)
        return torch.cat(feats), torch.tensor(labels)

    def update_bank(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        self.bank.update(feats.cpu(), labels.cpu())

    def bytes(self) -> int:
        return self.bank.bytes()

# ============================================================================
# Experience-replay buffer ---------------------------------------------------


class ERBuffer:
    """Reservoir buffer obeying a byte budget (stores fp16 images)."""

    def __init__(self, byte_budget: int, img_shape: Tuple[int, int, int] = (3, 32, 32)):
        self.byte_budget = byte_budget
        self.img_bytes = math.prod(img_shape) * 2  # fp16
        self.max_items = byte_budget // self.img_bytes
        self.storage: List[Tuple[torch.Tensor, int]] = []
        self.n_seen: int = 0

    # ---------------------------------------------------------------- add
    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for img, lbl in zip(x, y):
            self.n_seen += 1
            if len(self.storage) < self.max_items:
                self.storage.append((img.clone().half(), int(lbl)))
            else:
                j = random.randrange(self.n_seen)
                if j < self.max_items:
                    self.storage[j] = (img.clone().half(), int(lbl))

    # ---------------------------------------------------------------- sample
    def sample(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = random.sample(range(len(self.storage)), k)
        imgs, lbls = zip(*(self.storage[i] for i in idx))
        return torch.stack(list(imgs)).float(), torch.tensor(lbls)

    def bytes(self) -> int:  # noqa: D401
        return self.max_items * self.img_bytes

# ============================================================================
# Training loop --------------------------------------------------------------


def train_stream(
    backbone: nn.Module,
    clf: ExpandingClassifier,
    strategy: str,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    losr: Optional[LOSRMemory] = None,
    buffer: Optional[ERBuffer] = None,
    device: str = "cuda",
) -> None:
    """One pass over *loader* using *strategy* (LOSR / ER / NONE)."""

    backbone.train()
    clf.train()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # classifier must have enough heads for the current batch
        clf.ensure_capacity(int(y.max()) + 1)

        feats = backbone(x)
        loss = F.cross_entropy(clf(feats), y)

        # ------------------------------------------------------- replay logic
        if strategy == "LOSR" and losr is not None and losr.bank.mu_q:
            syn_f, syn_y = losr.generate()
            syn_f, syn_y = syn_f.to(device), syn_y.to(device)
            clf.ensure_capacity(int(syn_y.max()) + 1)
            loss = loss + F.cross_entropy(clf(syn_f), syn_y)

            eps = torch.randn(len(syn_y), 8, device=device)
            gen_f = losr.synth(eps, syn_y)
            loss = loss + F.cross_entropy(clf(gen_f), syn_y)

        elif strategy == "ER" and buffer is not None and len(buffer.storage) >= len(y):
            bx, by = buffer.sample(len(y))
            bx, by = bx.to(device), by.to(device)
            clf.ensure_capacity(int(by.max()) + 1)
            loss = loss + F.cross_entropy(clf(backbone(bx)), by)

        # ------------------------------------------------------- optimisation
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        # ------------------------------------------------------- memory update
        if strategy == "LOSR" and losr is not None:
            losr.update_bank(feats.detach(), y.detach())
        if strategy == "ER" and buffer is not None:
            buffer.add_batch(x.cpu(), y.cpu())
