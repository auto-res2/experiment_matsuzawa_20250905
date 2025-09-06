"""src/train.py – helpers for concept mining, causal generation and GC-DRO loss
NOTE:  compact, self-contained reference implementations that are *good enough*
for the refactored experiments to run.  They are **not** the full CGSI       
research code published in the paper.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image


# ---------------------------------------------------------------------------
#                             Concept Miner
# ---------------------------------------------------------------------------
class ConceptMiner:
    """Very small-scale concept extractor using ImageNet features + k-means.

    It is *highly simplified* but exposes the API needed by the experiments:
        • fit(dataloader, max_images)
        • rank_by_mutual_information(dataloader, labels)
    """

    def __init__(self, device: str = "cpu", num_clusters: int = 32):
        self.device = torch.device(device)
        self.num_clusters = num_clusters
        # Re-use a lightweight feature extractor (frozen) --------------------
        backbone = models.mobilenet_v2(pretrained=True)
        backbone.classifier = nn.Identity()
        backbone.eval().requires_grad_(False)
        self.backbone = backbone.to(self.device)
        self.kmeans: MiniBatchKMeans | None = None

    # .............................................
    @torch.inference_mode()
    def _embed_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        if imgs.device != self.device:
            imgs = imgs.to(self.device, non_blocking=True)
        feats = self.backbone(imgs)
        return feats.detach().cpu()

    def fit(self, dataloader, max_images: int = 2048):
        embeds: List[torch.Tensor] = []
        seen = 0
        for batch, _ in dataloader:
            embeds.append(self._embed_batch(batch))
            seen += len(batch)
            if seen >= max_images:
                break
        X = torch.cat(embeds).numpy()
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, batch_size=512)
        self.kmeans.fit(X)

    # .............................................
    def rank_by_mutual_information(self, dataloader, labels: List[int]):
        """Computes a *very crude* MI proxy: counts cluster/label co-occurrence."""
        if self.kmeans is None:
            raise RuntimeError("ConceptMiner must be .fit()-ed before ranking")
        assign_counts = np.zeros((self.num_clusters, max(labels) + 1), dtype=int)
        idx = 0
        for batch, _ in dataloader:
            embeds = self._embed_batch(batch).numpy()
            clust = self.kmeans.predict(embeds)
            for c in clust:
                assign_counts[c, labels[idx]] += 1
                idx += 1
        # Mutual information proxy – per-cluster purity ----------------------
        purity = assign_counts.max(axis=1) / assign_counts.sum(axis=1).clip(min=1)
        return purity.argsort()[::-1].tolist()


# ---------------------------------------------------------------------------
#                       Causal counterfactual generator
# ---------------------------------------------------------------------------
class CausalGenerator:
    """Placeholder that simply returns the original image (no diffusion).

    The *interface* matches the full CGSI implementation so that the higher-
    level experiment code remains unchanged.
    """

    def __init__(self, device: str = "cpu", model_name: str | None = None):
        self.device = device
        self.model_name = model_name  # logged for completeness only

    def generate_cf(self, img, mask):  # noqa: D401 – short name OK
        # Real implementation would call Stable Diffusion in-painting. Here we
        # just return the untouched image so that the pipeline runs offline.
        return img


# ---------------------------------------------------------------------------
#                             GC-DRO  loss  stub
# ---------------------------------------------------------------------------
class GCDROLoss(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, weights=None):  # noqa: D401 – pydoc
        if weights is None:
            return self.ce(logits, labels)
        if not torch.is_tensor(weights):
            weights = torch.tensor(weights, device=logits.device, dtype=torch.float32)
        weights = weights / weights.sum()
        loss = self.ce(logits, labels)
        return loss * (1 + self.alpha * weights.mean())
