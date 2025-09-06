"""
train.py – all model- and training-related utilities that are reused by
several experiments.  No experiment-specific logic is contained here.
"""
from __future__ import annotations

import os
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
    from PIL import Image  # noqa: F401 – imported for type-annotations only
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

    The heavyweight DINO-v2 backbone download/initialisation is *deferred* until
    it is *actually* required (first call to :py:meth:`fit` or
    :py:meth:`concept_presence`).  This keeps CI test-runs lightweight: merely
    constructing the class no longer triggers a >1 GB checkpoint download.
    """

    def __init__(self, device: str, num_clusters: int = 64):
        # Gracefully fall back to CPU if CUDA was requested but is unavailable.
        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available – falling back to CPU")
            device = "cpu"
        self.device = device
        self.num_clusters = num_clusters

        # Pick a safe default precision: float16 on GPU, float32 on CPU.
        self._feat_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        # Heavy objects will be instantiated lazily
        self.backbone: nn.Module | None = None

        self.kmeans: KMeans | None = None
        self.mi_scores_: list[float] | None = None

    # ------------------------------------------------------------------
    def _ensure_backbone(self):
        """Load the DINO-v2 backbone the first time it is requested."""
        if self.backbone is None:
            # The environment variable allows power-users to specify a *local*
            # path or a different Torch-hub stub, yet keeps defaults unchanged.
            repo = os.getenv("CGSI_DINOV2_REPO", "facebookresearch/dinov2")
            ckpt = os.getenv("CGSI_DINOV2_MODEL", "dinov2_vits14")
            print(f"[Info ] Loading DINO-v2 backbone {ckpt} from {repo} …")
            self.backbone = torch.hub.load(repo, ckpt, pretrained=True)
            self.backbone.eval().to(self.device)
            if hasattr(self.backbone, "head"):
                self.backbone.head = nn.Identity()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _tokens_from_loader(
        self, dl: DataLoader, *, max_images: int | None = None
    ) -> np.ndarray:  # noqa: D401,E501 – simple style OK
        """Stack patch-tokens for at most *max_images* from *dl* into an ndarray."""
        self._ensure_backbone()
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

    # ------------------------------------------------------------------
    def fit(self, dl: DataLoader, *, max_images: int | None = None):
        self._ensure_backbone()
        tokens = self._tokens_from_loader(dl, max_images=max_images)
        self.kmeans = KMeans(
            n_clusters=self.num_clusters, random_state=0, n_init="auto"
        ).fit(tokens)
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def concept_presence(self, img: torch.Tensor) -> np.ndarray:
        """Return a normalised histogram over cluster IDs for *one* image."""
        if self.kmeans is None:
            raise RuntimeError(
                "ConceptMiner must be .fit() before calling .concept_presence()."
            )
        self._ensure_backbone()
        out = self.backbone.get_intermediate_layers(
            img.unsqueeze(0).to(self.device, dtype=self._feat_dtype), n=1
        )[0]
        out = out[:, 1:, :].cpu().float().numpy().reshape(-1, out.shape[-1])
        labels = self.kmeans.predict(out)
        hist = np.bincount(labels, minlength=self.num_clusters) / len(labels)
        return hist

    # ------------------------------------------------------------------
    def rank_by_mutual_information(self, dl: DataLoader, labels: np.ndarray) -> list[int]:
        """Compute \hat{I}(C_k; Y) and return cluster indices sorted descending."""
        if self.kmeans is None:
            raise RuntimeError("ConceptMiner must be .fit() before ranking by MI.")
        hists = [self.concept_presence(img) for imgs, _ in dl for img in imgs]
        hists = np.stack(hists)
        mi_scores = [
            mutual_info_score(labels[: len(hists)], hists[:, k])
            for k in range(self.num_clusters)
        ]
        self.mi_scores_ = mi_scores
        return list(np.argsort(mi_scores)[::-1])


# =============================================================================
#            COUNTERFACTUAL  GENERATOR  (Stable-Diffusion-XL In-painting)
# =============================================================================
class CausalGenerator:
    """Thin wrapper around the SD-XL in-painting pipeline used by the project.

    Instantiation no longer forces an immediate ~6 GB checkpoint download.
    The pipeline is loaded lazily on the first call to
    :py:meth:`generate_cf`, making plain object creation inexpensive.
    """

    def __init__(self, device: str, model_name: str):
        # Resolve dtype / weight variant: fp16 only makes sense on CUDA.
        if device.startswith("cuda") and torch.cuda.is_available():
            self._pipe_kwargs = dict(torch_dtype=torch.float16, variant="fp16")
        else:
            # CPU inference needs full-precision weights.
            self._pipe_kwargs = dict(torch_dtype=torch.float32)
            if device.startswith("cuda") and not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available – loading SD-XL on CPU.")
            device = "cpu"

        self.device = device
        self._model_name = model_name
        self._pipe: StableDiffusionXLInpaintPipeline | None = None

    # ------------------------------------------------------------------
    def _ensure_pipe(self):
        """Instantiate the diffusers pipeline on-demand."""
        if self._pipe is None:
            print(f"[Info ] Loading SD-XL in-painting pipeline: {self._model_name} …")
            self._pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                self._model_name, **self._pipe_kwargs
            ).to(self.device)
            # Enable CPU off-load even on GPU builds – avoids running out of VRAM
            # in constrained test environments.
            self._pipe.enable_model_cpu_offload()

    # ------------------------------------------------------------------
    def generate_cf(
        self,
        img: "Image.Image",
        mask: "Image.Image",
        prompt: str = "",
    ) -> "Image.Image":
        """Generate a counterfactual by in-painting *mask* on *img*."""
        self._ensure_pipe()
        out = self._pipe(
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
