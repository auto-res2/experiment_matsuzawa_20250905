"""
train.py – all model- and training-related utilities that are reused by
several experiments.  No experiment-specific logic is contained here.
"""
from __future__ import annotations

import warnings
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

try:
    from diffusers import StableDiffusionXLInpaintPipeline
    from PIL import Image
except ModuleNotFoundError as e:  # fail fast – these libs are mandatory
    raise RuntimeError(
        "Required libraries for diffusion-based generation are missing."
    ) from e

__all__: List[str] = [
    "ConceptMiner",
    "CausalGenerator",
    "GCDROLoss",
]

# =============================================================================
#                     CONCEPT MINER  (DINO-v2  +  K-means)
# =============================================================================
class ConceptMiner:
    """Extract ViT patch tokens via DINO-v2 and cluster them with *K*-means.

    Only the minimal surface required by the experiments is implemented.  In
    particular we purposefully omit any batching logic for MI computation – the
    datasets involved are small, so a per-image loop keeps the code simple.
    """

    def __init__(self, device: str, num_clusters: int = 64):
        # If the caller asked for CUDA but no GPU is available we silently
        # downgrade to CPU rather than crashing – this is particularly useful
        # for CI environments where the config may still specify "cuda".
        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available – falling back to CPU")
            device = "cpu"
        self.device = device
        self.num_clusters = num_clusters

        # Pick a safe default precision: float16 on GPU, float32 on CPU.  Using
        # half precision on the CPU triggers many missing-kernel errors.
        self._feat_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        # ------------------------------------------------------------------
        # Load DINO-v2 backbone (remote weights cached locally by *torch.hub*).
        # ------------------------------------------------------------------
        self.backbone: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
        )
        self.backbone.eval().to(self.device)
        if hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()

        self.kmeans: KMeans | None = None
        self.mi_scores_: list[float] | None = None

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _tokens_from_loader(self, dl: DataLoader, *, max_images: int | None = None) -> np.ndarray:  # noqa: D401,E501
        """Stack patch-tokens for at most *max_images* from *dl* into an ndarray."""
        tokens, seen = [], 0
        for imgs, _ in dl:
            imgs = imgs.to(self.device, dtype=self._feat_dtype)
            out = self.backbone.get_intermediate_layers(imgs, n=1)[0]  # (B, 197, C)
            out = out[:, 1:, :].flatten(0, 1).cpu().float().numpy()  # (B*196, C)
            tokens.append(out)
            seen += imgs.size(0)
            if max_images and seen >= max_images:
                break
        return np.concatenate(tokens, axis=0)

    # ---------------------------------------------------------------------
    def fit(self, dl: DataLoader, *, max_images: int | None = None):
        tokens = self._tokens_from_loader(dl, max_images=max_images)
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init="auto").fit(tokens)
        return self

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def concept_presence(self, img: torch.Tensor) -> np.ndarray:
        """Return a normalised histogram over cluster IDs for *one* image."""
        if self.kmeans is None:
            raise RuntimeError("ConceptMiner must be .fit() before calling .concept_presence().")
        out = self.backbone.get_intermediate_layers(
            img.unsqueeze(0).to(self.device, dtype=self._feat_dtype), n=1
        )[0]
        out = out[:, 1:, :].cpu().float().numpy().reshape(-1, out.shape[-1])
        labels = self.kmeans.predict(out)
        hist = np.bincount(labels, minlength=self.num_clusters) / len(labels)
        return hist

    # ---------------------------------------------------------------------
    def rank_by_mutual_information(self, dl: DataLoader, labels: np.ndarray) -> list[int]:
        """Compute \hat{I}(C_k; Y) and return cluster indices sorted descending."""
        if self.kmeans is None:
            raise RuntimeError("ConceptMiner must be .fit() before ranking by MI.")
        hists = [self.concept_presence(img) for imgs, _ in dl for img in imgs]
        hists = np.stack(hists)
        mi_scores = [mutual_info_score(labels[: len(hists)], hists[:, k]) for k in range(self.num_clusters)]
        self.mi_scores_ = mi_scores
        return list(np.argsort(mi_scores)[::-1])


# =============================================================================
#            COUNTERFACTUAL  GENERATOR  (Stable-Diffusion-XL In-painting)
# =============================================================================
class CausalGenerator:
    """Thin wrapper around the SD-XL in-painting pipeline used by the project."""

    def __init__(self, device: str, model_name: str):
        # Resolve dtype / weight variant: fp16 only makes sense on CUDA.
        if device.startswith("cuda") and torch.cuda.is_available():
            pipe_kwargs = dict(torch_dtype=torch.float16, variant="fp16")
        else:
            # CPU inference needs full-precision weights.
            pipe_kwargs = dict(torch_dtype=torch.float32)
            if device.startswith("cuda") and not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available – loading SD-XL on CPU.")
            device = "cpu"

        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_name, **pipe_kwargs)
        self.pipe = self.pipe.to(device)
        # Enable CPU off-load even on GPU builds – avoids running out of VRAM in
        # constrained test environments.
        self.pipe.enable_model_cpu_offload()
        self.device = device

    # ------------------------------------------------------------------
    def generate_cf(self, img: "Image.Image", mask: "Image.Image", prompt: str = "") -> "Image.Image":
        """Generate a counterfactual by in-painting *mask* on *img*."""
        out = self.pipe(
            prompt=prompt,
            image=img,
            mask_image=mask,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=0.8,
        ).images[0]
        return out


# =============================================================================
#                       SOFT  GROUP-DRO  LOSS  (GC-DRO)
# =============================================================================
class GCDROLoss(nn.Module):
    """Continuous re-weighting variant of Group-Conditional DRO."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, g_vec: torch.Tensor):  # noqa: N802
        probs = g_vec / (g_vec.sum(dim=1, keepdim=True) + 1e-8)
        losses = F.cross_entropy(logits, targets, reduction="none")
        weighted = (probs * losses.unsqueeze(1)).sum(dim=1)
        return self.alpha * weighted.mean()
