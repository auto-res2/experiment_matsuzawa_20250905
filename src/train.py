"""src/train.py
All training-related utilities, model factory, algorithmic trainers, and the
counterfactual image generator live here.  Other modules should import only the
public symbols defined in __all__.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models as tv_models

import timm  # Vision-Transformers

try:
    from diffusers import StableDiffusionPipeline
except ImportError:  # optional dependency – handled gracefully
    StableDiffusionPipeline = None

# -----------------------------------------------------------------------------
# GLOBALS & LOGGER -------------------------------------------------------------
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s", datefmt="%H:%M:%S")
LOGGER = logging.getLogger("pcd.train")

# -----------------------------------------------------------------------------
# SMALL HELPER UTILITIES -------------------------------------------------------
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    """Returns the appropriate torch.device (GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# MODEL FACTORY ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_model(model_cfg: Dict[str, str | bool], num_classes: int) -> nn.Module:
    """Instantiates a classification model according to *model_cfg*."""
    if model_cfg["type"] == "resnet":
        model = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.DEFAULT if model_cfg.get("pretrained", False) else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_cfg["type"] == "vit":
        return timm.create_model("vit_small_patch16_224", pretrained=model_cfg.get("pretrained", False), num_classes=num_classes)
    raise ValueError(f"Unknown model type {model_cfg['type']}")


# -----------------------------------------------------------------------------
# COUNTERFACTUAL GENERATOR -----------------------------------------------------
# -----------------------------------------------------------------------------

class CounterfactualGenerator:
    """Light wrapper around Stable-Diffusion that produces edited images.

    The heavy LoRA fine-tuning should be carried out offline; we only load the
    resulting weights during runtime.  If *diffusers* is missing we raise early
    so the caller can fall back to non-PCD training.
    """

    def __init__(self, cfg: Dict):
        if StableDiffusionPipeline is None:
            raise RuntimeError("'diffusers' is not installed – CounterfactualGenerator unavailable.")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(get_device())
        self.pipe.enable_model_cpu_offload()

    @torch.no_grad()
    def generate(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:  # noqa: D401
        """Generate counterfactual images that match *images* in shape."""
        if len(prompts) != images.size(0):
            raise ValueError("Number of prompts must equal the batch size.")
        cf_tensors = []
        for prompt in prompts:
            out_img = self.pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
            cf_tensors.append(transforms.ToTensor()(out_img))
        return torch.stack(cf_tensors, dim=0).to(images.device, non_blocking=True)


# -----------------------------------------------------------------------------
# GENERIC TRAINER BASE CLASS ---------------------------------------------------
# -----------------------------------------------------------------------------

class BaseTrainer:
    """Common utilities for all algorithmic trainers."""

    def __init__(self, model: nn.Module, cfg: Dict, meta: Dict):
        self.model = model.to(get_device(), non_blocking=True)
        self.cfg = cfg
        self.meta = meta
        self.ce = nn.CrossEntropyLoss(reduction="none")

        self.optim = optim.AdamW(
            self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8
        )
        self.scheduler = CosineAnnealingLR(self.optim, T_max=cfg["epochs"], eta_min=1e-6)

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    # ------------------------------------------------------------------

    def fit(self, train_loader, val_loader):
        best_acc = 0.0
        history = {"train_loss": [], "val_acc": []}
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            running = 0.0
            for batch in train_loader:
                loss = self.training_step(batch)
                running += loss.item()
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optim.step()
            self.scheduler.step()

            val_acc = self.evaluate(val_loader)
            history["train_loss"].append(running / len(train_loader))
            history["val_acc"].append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), RESULTS_DIR / "best_model.pt")
            LOGGER.info(
                "Epoch %d/%d – loss=%.4f – val_acc=%.2f",
                epoch + 1,
                self.cfg["epochs"],
                running / len(train_loader),
                val_acc,
            )
        return best_acc, history

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        for x, y in data_loader:
            x = x.to(get_device(), non_blocking=True)
            y = y.to(get_device(), non_blocking=True)
            preds = self.model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return 100.0 * correct / total

    # ------------------------------------------------------------------
    # The child class MUST implement -----------------------------------
    # ------------------------------------------------------------------

    def training_step(self, batch):  # noqa: D401
        raise NotImplementedError


# -----------------------------------------------------------------------------
# ERM TRAINER ------------------------------------------------------------------
# -----------------------------------------------------------------------------

class ERMTrainer(BaseTrainer):
    def training_step(self, batch):
        x, y = batch[0].to(get_device(), non_blocking=True), batch[1].to(get_device(), non_blocking=True)
        return self.ce(self.model(x), y).mean()


# -----------------------------------------------------------------------------
# GROUP DRO TRAINER ------------------------------------------------------------
# -----------------------------------------------------------------------------

class GroupDROTrainer(BaseTrainer):
    """Implements the max-group loss from Sagawa et al., 2020."""

    def training_step(self, batch):
        x = batch[0].to(get_device(), non_blocking=True)
        y = batch[1].to(get_device(), non_blocking=True)
        g = batch[2].to(get_device(), non_blocking=True)

        losses = self.ce(self.model(x), y)  # (B,)
        group_losses = [losses[g == ug].mean() for ug in g.unique()]
        return torch.stack(group_losses).max()


# -----------------------------------------------------------------------------
# PCD TRAINER ------------------------------------------------------------------
# -----------------------------------------------------------------------------

class PCDTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, cfg: Dict, meta: Dict, generator: CounterfactualGenerator):
        super().__init__(model, cfg, meta)
        self.gen = generator
        self.lambda_inv = cfg["lambda_inv"]
        self.lambda_feat = cfg["lambda_feat"]

    def _extract_features(self, imgs: torch.Tensor):
        """Lightweight feature extractor that works for ResNets & timm ViTs."""
        if hasattr(self.model, "forward_features"):  # timm-models
            return self.model.forward_features(imgs)
        # torchvision ResNet fallback
        feats = self.model.relu(self.model.bn1(self.model.conv1(imgs)))
        feats = self.model.layer1(self.model.maxpool(feats))
        feats = self.model.layer4(self.model.layer3(self.model.layer2(feats)))
        feats = torch.flatten(self.model.avgpool(feats), 1)
        return feats

    def training_step(self, batch):
        x, y = batch[0].to(get_device(), non_blocking=True), batch[1].to(get_device(), non_blocking=True)
        bsz = x.size(0)

        prompts = ["a photo with changed background" for _ in range(bsz)]  # placeholder prompts
        x_cf = self.gen.generate(x, prompts)

        imgs_all = torch.cat([x, x_cf], dim=0)  # 2B, C, H, W
        logits_all = self.model(imgs_all)
        feats_all = self._extract_features(imgs_all)

        logits_orig, logits_cf = logits_all[:bsz], logits_all[bsz:]
        feat_orig, feat_cf = feats_all[:bsz], feats_all[bsz:]

        loss_erm = self.ce(logits_orig, y).mean()
        loss_inv = self.ce(logits_cf, y).mean()
        loss_feat = F.mse_loss(feat_orig, feat_cf)

        return loss_erm + self.lambda_inv * loss_inv + self.lambda_feat * loss_feat


__all__ = [
    "get_device",
    "build_model",
    "CounterfactualGenerator",
    "BaseTrainer",
    "ERMTrainer",
    "GroupDROTrainer",
    "PCDTrainer",
]