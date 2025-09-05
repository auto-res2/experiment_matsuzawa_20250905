"""src/train.py
Model, replay-memory and training logic for HCER experiments.
This file concentrates everything related to the forward / backward
passes so that src.main can remain a thin orchestration layer.
"""
from __future__ import annotations
import random, math, collections, pathlib, json, time
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import GradScaler, autocast

# -----------------------------------------------------------------------------
#                      Reproducibility helper
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0):
    """Reproduce the same results by fixing all random sources."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
#                      Backbone factory
# -----------------------------------------------------------------------------

_BACKBONES = {
    "resnet18": lambda: torchvision.models.resnet18(weights=None),
    "mobilenet_v3_small": lambda: torchvision.models.mobilenet_v3_small(weights=None),
}


def get_backbone(name: str) -> Tuple[nn.Module, int]:
    if name not in _BACKBONES:
        raise KeyError(f"Unsupported backbone: {name}")
    net = _BACKBONES[name]()
    if name.startswith("resnet"):
        in_features = net.fc.in_features
        net.fc = nn.Identity()
    elif name.startswith("mobilenet"):
        in_features = net.classifier[-1].in_features
        net.classifier = nn.Identity()
    else:
        raise RuntimeError("Unknown backbone modification required")
    return net, in_features

# -----------------------------------------------------------------------------
#                      Replay-memory implementations
# -----------------------------------------------------------------------------

class BaseMemory:
    """Abstract replay buffer interface."""
    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes

    def sample(self, n: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError

    def footprint_mb(self) -> float:
        return self.budget_bytes / 2 ** 20


# 1)  Naïve episodic replay ----------------------------------------------------
class ERMemory(BaseMemory):
    """Reservoir buffer that stores raw images."""
    def __init__(self, img_shape: tuple[int, int, int], budget_bytes: int):
        super().__init__(budget_bytes)
        self.img_shape = img_shape
        self.capacity = budget_bytes // (img_shape[0] * img_shape[1] * img_shape[2])
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.n_seen = 0

    def sample(self, n: int):
        if not self.images:
            return {}
        idx = random.sample(range(len(self.images)), k=min(n, len(self.images)))
        return {
            "x": torch.stack([self.images[i] for i in idx]),
            "y": torch.tensor([self.labels[i] for i in idx]),
        }

    def update(self, x: torch.Tensor, y: torch.Tensor):
        for xi, yi in zip(x, y):
            self.n_seen += 1
            if len(self.images) < self.capacity:
                self.images.append(xi.cpu())
                self.labels.append(int(yi))
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.capacity:
                    self.images[j] = xi.cpu()
                    self.labels[j] = int(yi)


# 2)  HCER memory --------------------------------------------------------------
class HCERMemory(BaseMemory):
    """Hierarchically-Compressed Episodic Replay (Tier-1 + Tier-2)."""
    def __init__(
        self,
        feature_dim: int,
        tier1_bits: int = 64,
        anchors_per_class: int = 2,
        num_classes: int = 100,
        budget_bytes: int = 10 * 2 ** 20,
    ):
        super().__init__(budget_bytes)
        # Tier-1 PQ parameters ------------------------------------------------
        self.subspaces = 4
        self.centres = 16  # => 4 bits per subcode
        if tier1_bits not in {32, 48, 64, 128}:
            raise ValueError("Unsupported code length")
        self.tier1_bits = tier1_bits
        self.sub_dim = feature_dim // self.subspaces
        self.codebook = torch.zeros(self.subspaces, self.centres, self.sub_dim)
        torch.nn.init.normal_(self.codebook, std=0.02)
        self.codes: Dict[int, List[int]] = collections.defaultdict(list)

        # Tier-2 simple MLP decoder ----------------------------------------
        self.anchors_per_class = anchors_per_class
        self.latent_dim = 256
        img_dim = 3 * 32 * 32
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Sigmoid(),
        )
        self.tier2_latents: Dict[int, List[torch.Tensor]] = collections.defaultdict(list)

        # Promotion bookkeeping -------------------------------------------
        self.k_promote = 3
        self.loss_window: List[float] = []
        self.example_window: List[Tuple[torch.Tensor, int]] = []

    # ---------------- private helpers ---------------------------------------
    def _encode_feature(self, feat: torch.Tensor) -> int:
        """Product-quantise a feature vector into a compact integer."""
        codes = []
        for s in range(self.subspaces):
            sub_f = feat[..., s * self.sub_dim : (s + 1) * self.sub_dim]
            dists = (self.codebook[s] - sub_f).pow(2).sum(-1)
            idx = int(torch.argmin(dists))
            codes.append(idx)
        packed = 0
        for i, c in enumerate(codes):
            packed |= (c & 0xF) << (i * 4)
        return packed

    def hebbian_update(self, feat: torch.Tensor, codes: List[int]):
        eta = 0.05
        for s, idx in enumerate(codes):
            self.codebook[s, idx] = (1 - eta) * self.codebook[s, idx] + eta * feat[
                ..., s * self.sub_dim : (s + 1) * self.sub_dim
            ]

    # ---------------- API methods ------------------------------------------
    def sample(self, n: int):
        n_t1 = math.ceil(n * 0.9)
        n_t2 = n - n_t1
        xs, ys = [], []
        # Tier-2 (decoded anchors) --------------------------------------
        if n_t2 > 0 and any(self.tier2_latents.values()):
            for _ in range(n_t2):
                cls = random.choice(list(self.tier2_latents.keys()))
                latent = random.choice(self.tier2_latents[cls])
                img = self.decoder(latent.to(self.decoder[0].weight.device)).view(3, 32, 32)
                xs.append(img)
                ys.append(cls)
        # Tier-1 (binary codes) -----------------------------------------
        if self.codes:
            for _ in range(n_t1):
                cls = random.choice(list(self.codes.keys()))
                code_int = random.choice(self.codes[cls])
                # very cheap hallucination around PQ centres --------------
                centres = []
                for s in range(self.subspaces):
                    sub_idx = (code_int >> (s * 4)) & 0xF
                    centres.append(self.codebook[s, sub_idx])
                feat = torch.cat(centres) + 0.05 * torch.randn_like(torch.cat(centres))
                img = torch.clamp(torch.randn(3, 32, 32) * 0.1 + 0.5, 0, 1)
                xs.append(img)
                ys.append(cls)
        if not xs:
            return {}
        return {"x": torch.stack(xs), "y": torch.tensor(ys)}

    def update(self, *, features: torch.Tensor, labels: torch.Tensor, losses: torch.Tensor):
        # Update Tier-1 codes & codebook --------------------------------
        for f, y in zip(features, labels):
            code_int = self._encode_feature(f)
            self.codes[int(y)].append(code_int)
            self.hebbian_update(f, [(code_int >> (i * 4)) & 0xF for i in range(self.subspaces)])

        # Loss-aware promotion -----------------------------------------
        for f, y, l in zip(features, labels, losses):
            self.example_window.append((f.detach(), int(y)))
            self.loss_window.append(float(l))
        if len(self.loss_window) >= 50:
            topk = sorted(range(len(self.loss_window)), key=lambda i: self.loss_window[i], reverse=True)[: self.k_promote]
            for idx in topk:
                feat, cls = self.example_window[idx]
                if len(self.tier2_latents[cls]) < self.anchors_per_class:
                    latent = torch.randn(self.latent_dim)
                    self.tier2_latents[cls].append(latent)
            self.loss_window.clear()
            self.example_window.clear()

# -----------------------------------------------------------------------------
#                      Trainer class
# -----------------------------------------------------------------------------
class ContinualTrainer:
    """End-to-end continual-learning loop (single GPU)."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        device: str = "cuda",
        memory: str = "hcer",
        budget_mb: int = 10,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        backbone, feat_dim = get_backbone(model_name)
        self.backbone = backbone.to(self.device)
        self.fc = nn.Linear(feat_dim, num_classes).to(self.device)
        self.model_name = model_name
        self.loss_fn = nn.CrossEntropyLoss()
        self.scaler = GradScaler()

        # memory -------------------------------------------------------
        budget_bytes = budget_mb * 2 ** 20
        if memory == "er":
            self.memory = ERMemory(img_shape=(32, 32, 3), budget_bytes=budget_bytes)
        elif memory == "hcer":
            self.memory = HCERMemory(
                feature_dim=feat_dim,
                tier1_bits=64,
                anchors_per_class=2,
                num_classes=num_classes,
                budget_bytes=budget_bytes,
            )
        else:
            raise NotImplementedError(memory)

        self.opt = torch.optim.SGD(
            list(self.backbone.parameters()) + list(self.fc.parameters()),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
        )

    # -----------------------------------------------------------------
    def _forward(self, x):
        feat = self.backbone(x)
        out = self.fc(feat)
        return out, feat

    # -----------------------------------------------------------------
    def observe_task(self, train_loader, *, epochs: int = 1):
        self.backbone.train(); self.fc.train()
        for _ in range(epochs):
            for batch in train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                # add replay ------------------------------------------------
                replay = self.memory.sample(n=x.size(0) // 2)
                if replay:
                    rx, ry = replay["x"].to(self.device), replay["y"].to(self.device)
                    x = torch.cat([x, rx], 0)
                    y = torch.cat([y, ry], 0)
                self.opt.zero_grad(set_to_none=True)
                with autocast():
                    logits, feats = self._forward(x)
                    loss = self.loss_fn(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                # memory update --------------------------------------
                with torch.no_grad():
                    losses_per_sample = nn.functional.cross_entropy(logits, y, reduction="none")
                    self.memory.update(
                        features=feats.detach().cpu(),
                        labels=y.detach().cpu(),
                        losses=losses_per_sample.detach().cpu(),
                    )

    # -----------------------------------------------------------------
    def evaluate(self, test_loader) -> float:
        self.backbone.eval(); self.fc.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self._forward(x)
                pred = torch.argmax(logits, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        self.backbone.train(); self.fc.train()
        return 100.0 * correct / total
