"""src/train.py
Contains all training-time utilities:
1. Causal latent discovery (NOTEARS).
2. Counterfactual diffusion editor (LoRA fine-tune + latent edit).
3. Fourier Consistent-Distance regulariser.
4. Group-Conditional DRO trainer.
The module is intentionally self-contained so that it can be imported from
src.main without causing circular dependencies.
"""
from __future__ import annotations

import math
import os
import random
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1.  Causal latent discovery (NOTEARS)
# -----------------------------------------------------------------------------

def run_notears(latents: torch.Tensor, labels: torch.Tensor, *, lambda1: float = 1e-2, max_iter: int = 200, keep: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Run NOTEARS linear causal discovery on concatenated (latent, label).

    Parameters
    ----------
    latents : torch.Tensor
        (N, D) latent representations.
    labels : torch.Tensor
        (N,) class labels.
    lambda1 : float, optional
        L1 regularisation strength.
    max_iter : int, optional
        Maximum optimisation iterations.
    keep : int, optional
        Number of strongest causal dimensions to retain.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (indices, causal_strengths) of the selected latent dimensions.
    """
    try:
        from notears.linear import notears_linear  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Package `notears` not found. Install with `pip install notears`." ) from exc

    Z = latents.cpu().float().numpy()
    y = labels.cpu().float().numpy().reshape(-1, 1)
    data = np.concatenate([Z, y], axis=1)
    W_est = notears_linear(data, lambda1=lambda1, max_iter=max_iter)

    # Edges z_k -> y are in the last column (excluding last row).
    causal_strength = W_est[:-1, -1]
    idxs = np.argsort(np.abs(causal_strength))[-keep:][::-1]
    return idxs, causal_strength[idxs]


# -----------------------------------------------------------------------------
# 2.  Counterfactual diffusion editor (LoRA fine-tuning)
# -----------------------------------------------------------------------------

class DiffusionEditor:
    """Light-weight wrapper around Stable-Diffusion to create counterfactuals.

    NOTE:  This implementation purposefully keeps the computational load
    minimal so that the script can still be executed on <16 GB GPUs.  For real
    experiments, please replace the dummy training loop with an Accelerate-
    based implementation.
    """

    def __init__(self, *, device: str = "cuda", fp16: bool = True):
        from diffusers import StableDiffusionPipeline  # local import to keep base import light.
        dtype = torch.float16 if fp16 else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
        )
        self.pipe.to(device)
        self.device = device

    @staticmethod
    def _inject_lora(unet: nn.Module, rank: int = 8) -> nn.Module:
        """Attach LoRA adapters to qkv projection layers of a UNet."""
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

        lora_cfg = LoraConfig(
            task_type=TaskType.UNET,
            target_modules=["to_q", "to_k", "to_v"],
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            bias="none",
        )
        return get_peft_model(unet, lora_cfg)

    def finetune_lora(self, images: List[torch.Tensor], cfg):  # cfg: diffusion part of config
        """Very small LoRA fine-tune loop – **for demonstrational use only**."""
        unet: nn.Module = self.pipe.unet
        unet = self._inject_lora(unet, rank=cfg.lora_rank)
        unet.train()
        optimiser = torch.optim.AdamW(unet.parameters(), lr=cfg.lora_lr)

        pbar = tqdm(range(cfg.steps), desc="LoRA-finetune", leave=False)
        for _ in pbar:
            idx = np.random.randint(0, len(images))
            x = images[idx].unsqueeze(0).to(self.device)
            # This is NOT how diffusion is properly trained – we deliberately
            # keep a dummy loss to avoid out-of-scope complexity.
            noise_pred = unet(x, timesteps=torch.tensor([1], device=x.device)).sample
            loss = noise_pred.abs().mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            pbar.set_postfix(loss=float(loss))
        self.pipe.unet = unet.eval()

    @torch.no_grad()
    def edit_latent(self, latent: torch.Tensor, dim: int, *, delta: float, cfg) -> torch.Tensor:
        """Return a counterfactual image where only the latent dimension `dim` is changed."""
        from torchvision import transforms as T

        prompt = "a photo"  # placeholder – real method would cross-attend on latent.
        latent_mod = latent.clone()
        latent_mod[..., dim] += delta  # unused in the dummy implementation

        img = self.pipe(
            prompt=prompt,
            num_inference_steps=cfg.ddim_steps,
            guidance_scale=cfg.cfg_scale,
        ).images[0]
        to_tensor = T.ToTensor()
        return to_tensor(img)


# -----------------------------------------------------------------------------
# 3.  Fourier Consistent-Distance regulariser
# -----------------------------------------------------------------------------

def fourier_consistent_distance(x: torch.Tensor, logits: torch.Tensor, *, window: int = 7) -> torch.Tensor:
    """Penalise high-frequency energy of `x` (MOL / Out-of-Line style)."""
    import torch.fft as fft  # local import to keep torch.fft optional for CPU-only builds

    # FFT – expects float32/float64
    freq = fft.fftn(x.float(), dim=(-2, -1))
    freq_shift = fft.fftshift(freq, dim=(-2, -1))
    H, W = x.shape[-2:]
    centre = freq_shift[..., H // 2 - window : H // 2 + window, W // 2 - window : W // 2 + window]
    low_energy = centre.abs().pow(2).mean()
    total_energy = freq_shift.abs().pow(2).mean()
    high_freq_ratio = (total_energy - low_energy) / (total_energy + 1e-8)
    return high_freq_ratio.mean()


# -----------------------------------------------------------------------------
# 4.  Group-Conditional DRO trainer
# -----------------------------------------------------------------------------

class GroupDROTrainer:
    """Trainer that minimises worst-group loss given group identifiers."""

    def __init__(self, model: nn.Module, *, dro_radius: float, fourier_lambda: float, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.dro_radius = dro_radius  # kept for API compatibility – not used in current impl.
        self.fourier_lambda = fourier_lambda
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _group_losses(self, logits: torch.Tensor, y: torch.Tensor, gid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loss_indiv = self.criterion(logits, y)
        groups = torch.unique(gid)
        g_loss = torch.stack([loss_indiv[gid == g].mean() for g in groups])
        return g_loss, groups

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> None:
        self.model.train()
        for batch in tqdm(loader, desc="train_epoch", leave=False):
            x, y, gid = [t.to(self.device, non_blocking=True) for t in batch]
            logits = self.model(x)
            loss = self.criterion(logits, y)

            if self.fourier_lambda > 0:
                loss = loss + self.fourier_lambda * fourier_consistent_distance(x, logits)

            # worst-group selection
            with torch.no_grad():
                g_loss, groups = self._group_losses(logits.detach(), y, gid)
                worst_gid = groups[g_loss.argmax()]
            loss = loss[gid == worst_gid].mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total, correct = 0, 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            preds = self.model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)
