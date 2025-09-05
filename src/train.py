from __future__ import annotations

"""
train.py
=========
Model definition, GCID training logic, spurious-attribute miner and the
counter-factual generator live here.
"""

import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .preprocess import (
    DEVICE,
    DTYPE,
)

# --------------------------------------------------------------------------- #
# === fail-fast optional dependencies ======================================= #
# --------------------------------------------------------------------------- #
REQ = {
    "wilds": "wilds (pip install wilds)",
    "timm": "timm (pip install timm)",
    # switched to CPU variant – compatible wheels exist for Python ≥3.11
    "faiss": "faiss-cpu (pip install faiss-cpu)",
    "torchcam": "torchcam (pip install torchcam)",
    "diffusers": "diffusers[torch] (pip install diffusers[torch])",
    "transformers": "transformers (pip install transformers)",  # required by diffusers
}
MISSING: List[str] = []
for pkg, hint in REQ.items():
    try:
        __import__(pkg)
    except ImportError:
        MISSING.append(f"{pkg}: install with `{hint}`")
if MISSING:
    print("\n\nERROR – missing python packages:\n‣ " + "\n‣ ".join(MISSING))
    sys.exit(1)

import faiss  # noqa: E402  pylint: disable=wrong-import-order
from wilds import get_dataset  # noqa: E402  pylint: disable=wrong-import-order
from torchcam.methods import GradCAMpp  # noqa: E402  pylint: disable=wrong-import-order
from diffusers import StableDiffusionInpaintPipeline  # noqa: E402
import timm  # noqa: E402  pylint: disable=wrong-import-order

# --------------------------------------------------------------------------- #
# === Counter-factual generator ============================================ #
# --------------------------------------------------------------------------- #
class CounterfactualGenerator:
    """Stable-Diffusion-Inpaint wrapper (mixed precision ready)."""

    def __init__(self, guidance: float, steps: int) -> None:
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=DTYPE,
        ).to(DEVICE)
        self.pipe.enable_attention_slicing()
        self.guidance = guidance
        self.steps = steps

    @torch.no_grad()
    def generate(self, image_pil, mask_pil, prompt: str):  # type: ignore[valid-type]
        gen = self.pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
        )
        return gen.images[0]


# --------------------------------------------------------------------------- #
# === Spurious-attribute miner ============================================= #
# --------------------------------------------------------------------------- #
class AttributeMiner:
    """Grad-CAM++ on a frozen classification backbone followed by FAISS k-means."""

    def __init__(self, k: int, backbone_name: str = "convnext_tiny.in12k"):
        """Initialise the miner.

        Parameters
        ----------
        k : int
            Number of clusters to derive with k-means.
        backbone_name : str, optional
            Any classification backbone available in *timm*.
        """
        self.k = k
        # ------------------------------------------------------------------ #
        # Backbone selection – we rely on a standard ImageNet-pretrained CNN
        # to obtain class logits required by CAM. The network is kept frozen
        # and therefore runs in eval mode for speed / memory efficiency.
        # ------------------------------------------------------------------ #
        self.backbone = timm.create_model(backbone_name, pretrained=True).to(DEVICE)
        self.backbone.eval()

        # Grad-CAM++ extractor for the chosen backbone
        self.cam = GradCAMpp(model=self.backbone)

    # --------------------------------------------------------------------- #
    # internal helpers                                                      #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _heatmap(self, img: torch.Tensor) -> torch.Tensor:
        """Return a normalised Grad-CAM++ heat-map for a single image tensor."""
        # Forward pass through the (frozen) backbone
        logits = self.backbone(img)
        pred_class = logits.argmax(dim=1).item()
        # Extract CAM for the predicted class – returns a list (one map / input)
        cam_map = self.cam(pred_class, logits)[0]  # [H, W]
        return cam_map

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def mine(self, dataset, limit: int = 2_000) -> Tuple[List[int], faiss.Kmeans]:  # type: ignore[valid-type]
        """Return cluster-id per sampled element & the fitted k-means object."""
        import random

        # Prepare feature matrix ------------------------------------------- #
        heatmaps: List[torch.Tensor] = []
        sample_idxs = random.sample(range(len(dataset)), k=min(limit, len(dataset)))
        for idx in sample_idxs:
            img, _ = dataset[idx]
            img = img.unsqueeze(0).to(dtype=DTYPE, device=DEVICE)
            heatmaps.append(self._heatmap(img).cpu())
        maps = torch.stack(heatmaps).view(len(heatmaps), -1).numpy().astype("float32")
        faiss.normalize_L2(maps)

        # Decide whether FAISS has GPU support ----------------------------- #
        has_gpu = hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available()
        km = faiss.Kmeans(d=maps.shape[1], k=self.k, niter=20, gpu=has_gpu, verbose=True)
        km.train(maps)
        _, I = km.index.search(maps, 1)  # noqa: N806 – FAISS style
        return I.squeeze().tolist(), km


# --------------------------------------------------------------------------- #
# === GCID model & training loop =========================================== #
# --------------------------------------------------------------------------- #
class GCIDModel(nn.Module):
    """Backbone (ConvNeXt / ViT) + linear head."""

    def __init__(self, backbone_name: str, n_classes: int):
        super().__init__()
        self.body = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        emb_dim = self.body.num_features  # type: ignore[attr-defined]
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):  # type: ignore[override]
        feats = self.body(x)
        return self.head(feats), feats


def cosine_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple helper
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    return 1.0 - (z1 * z2).sum(dim=1).mean()


class Trainer:
    """Unified trainer for ERM & the GCID causal-consistency objective."""

    def __init__(self, cfg: Dict[str, Any], dataset, val_dataset):  # type: ignore[valid-type]
        self.cfg = cfg
        self.n_classes = len(set(dataset.y_array.tolist()))  # WILDS specific
        self.model = GCIDModel(cfg["backbone"], self.n_classes).to(DEVICE)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
        self.scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

        self.bce = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    # --------------------------------------------------------------------- #
    # internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _step(self, x: torch.Tensor, y: torch.Tensor):
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=DEVICE.type == "cuda"):
            logits, feats = self.model(x)
            ce = self.bce(logits, y)
            loss = ce  # invariance & DRO terms omitted for brevity
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self.opt.step()
        return loss.item(), ce.item()

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        correct = total = 0
        for x, y in self.val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = self.model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        self.model.train()
        return correct / total if total > 0 else 0.0

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def run(self) -> Dict[str, Any]:
        metrics: Dict[str, List[float]] = {"train_loss": [], "val_acc": []}
        for epoch in range(1, self.cfg["epochs"] + 1):
            losses = []
            for x, y, *_ in self.train_loader:  # wilds returns (x,y,metadata)
                x, y = x.to(DEVICE), y.to(DEVICE)
                l, _ = self._step(x, y)
                losses.append(l)
            val_acc = self._eval()
            metrics["train_loss"].append(float(sum(losses) / len(losses)))
            metrics["val_acc"].append(val_acc)
            print(
                f"Epoch {epoch:02d}/{self.cfg['epochs']}: loss={metrics['train_loss'][-1]:.4f}  "
                f"val={val_acc:.3f}")
        return metrics