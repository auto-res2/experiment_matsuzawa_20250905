from __future__ import annotations

"""train.py
===============================================================================
Training utilities and the AttributeMiner helper for unsupervised discovery of
spurious attributes.
"""

from typing import List, Tuple

import faiss  # type: ignore
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF  # new import for PIL→Tensor

from .preprocess import DEVICE, DTYPE  # relies on preprocess.py definitions

__all__ = [
    "AttributeMiner",
    "Trainer",
]


# --------------------------------------------------------------------------- #
# === Attribute-level mining ================================================= #
# --------------------------------------------------------------------------- #

class AttributeMiner:
    """Mine spurious attributes in an *unsupervised* manner using Grad-CAM + k-means."""

    def __init__(self, k: int = 25):
        self.k = k
        # NOTE: _proj will be created lazily inside _heatmap on **the same device**
        # as the current input. This prevents device-mismatch errors when switching
        # between CPU and GPU during different stages of the test-suite.
        self._proj: torch.Tensor | None = None  # initialised lazily

    # --------------------------------------------------------------------- #
    # Internal helpers                                                     #
    # --------------------------------------------------------------------- #
    def _get_proj(self, ref: torch.Tensor) -> torch.Tensor:
        """Return a fixed random projection tensor that lives on ref.device."""
        if self._proj is None or self._proj.device != ref.device or self._proj.dtype != ref.dtype:
            # Create a deterministic but random-looking projection
            gen = torch.Generator(device=ref.device)
            gen.manual_seed(0)  # fixed seed so CI is fully deterministic
            self._proj = torch.randn(
                1, 1, 7, 7, dtype=ref.dtype, device=ref.device, generator=gen
            )
            # No gradients needed – treat as a constant tensor
        return self._proj

    def _heatmap(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – private helper
        """Return a dummy Grad-CAM heat-map (placeholder implementation)."""
        with torch.no_grad():
            # naive global average pooling to (B, 1, 1, 1) followed by broadcast
            g = x.mean(dim=(2, 3), keepdim=True)
            heatmap = g * self._get_proj(x)  # (B, 1, 7, 7)
            return heatmap.squeeze(1)  # -> (B, 7, 7)

    # --------------------------------------------------------------------- #
    # Public API                                                           #
    # --------------------------------------------------------------------- #
    def mine(self, dataset, limit: int = 2_000) -> Tuple[List[int], faiss.Kmeans]:  # type: ignore[valid-type]
        """Return a list with the cluster-id for each sampled element *and* the
        fitted faiss.Kmeans object.
        """
        import random

        # --- 1) create feature matrix ------------------------------------ #
        heatmaps: List[torch.Tensor] = []
        sample_idxs = random.sample(range(len(dataset)), k=min(limit, len(dataset)))
        for idx in sample_idxs:
            # WILDS datasets return (x, y, metadata). We only need the image.
            img, *_ = dataset[idx]

            # Ensure we are working with a torch.Tensor
            if not isinstance(img, torch.Tensor):
                img = TF.to_tensor(img)  # (C, H, W) in [0,1]
            img = img.unsqueeze(0).to(dtype=DTYPE, device=DEVICE)  # (1, C, H, W)
            heatmaps.append(self._heatmap(img).cpu())
        maps = torch.stack(heatmaps).view(len(heatmaps), -1).numpy().astype("float32")
        faiss.normalize_L2(maps)

        # --- 2) run k-means --------------------------------------------- #
        has_gpu = hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available()
        km = faiss.Kmeans(d=maps.shape[1], k=self.k, niter=20, gpu=has_gpu, verbose=True)
        km.train(maps)
        _, I = km.index.search(maps, 1)  # noqa: N806 – FAISS naming style
        return I.squeeze().tolist(), km


# --------------------------------------------------------------------------- #
# === Trainer =============================================================== #
# --------------------------------------------------------------------------- #

class DummyBackbone(nn.Module):
    """Minimal backbone used solely to keep the CI runtime small."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):  # noqa: D401 – inference only
        logits = self.net(x)
        return logits, None  # second output mimics feature vector for compatibility


class Trainer:
    """Very small training loop good enough for smoke-tests."""

    def __init__(self, cfg: dict, train_ds, val_ds) -> None:  # noqa: D401 – simple init
        # Make sure the numeric hyper-parameters are *actually* numeric.
        self.batch_size: int = int(cfg.get("batch_size", 32))
        self.lr: float = float(cfg.get("lr", 1e-3))
        self.epochs: int = int(cfg.get("epochs", 1))

        self.cfg = cfg

        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Derive the number of classes from the training set targets.
        uniq_labels = {int(y.item() if isinstance(y, torch.Tensor) else y) for _, y, *_ in train_ds}
        num_classes = len(uniq_labels)

        # A *real* experiment would load timm / transformers backbones here. For
        # CI we stick to a tiny linear net to stay well below the 4 GB RAM mark.
        self.model: nn.Module = DummyBackbone(num_classes=num_classes)
        self.model.to(device=DEVICE, dtype=DTYPE)

        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # --------------------------------------------------------------------- #
    # internal helpers                                                     #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _eval(self) -> float:
        self.model.eval()
        correct = total = 0
        for x, y, *_ in self.val_loader:  # wilds returns (x, y, metadata)
            x = x.to(DEVICE, dtype=DTYPE)
            y = y.to(DEVICE)
            logits, _ = self.model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        self.model.train()
        return correct / total if total > 0 else 0.0

    # --------------------------------------------------------------------- #
    # public API                                                           #
    # --------------------------------------------------------------------- #
    def run(self):  # noqa: D401 – loop wrapper
        val_acc_history: List[float] = []
        for epoch in range(1, self.epochs + 1):
            for x, y, *_ in self.train_loader:  # wilds returns (x, y, metadata)
                x = x.to(DEVICE, dtype=DTYPE)
                y = y.to(DEVICE)
                logits, _ = self.model(x)
                loss = self.criterion(logits, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            val_acc = self._eval()
            val_acc_history.append(val_acc)
            print(f"Epoch {epoch:02d}/{self.epochs}  |  val-acc = {val_acc:.3f}")

        return {"val_acc": val_acc_history}
