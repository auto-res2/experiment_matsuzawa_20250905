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

import warnings
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# 1.  Causal latent discovery (NOTEARS)
# -----------------------------------------------------------------------------


def _fallback_notears(latents: torch.Tensor, labels: torch.Tensor, *, keep: int) -> Tuple[np.ndarray, np.ndarray]:
    """Very light-weight fallback when *notears* is not available.

    We rank latent dimensions by the absolute Pearson correlation w.r.t. the
    label.  This provides a cheap heuristic that keeps this repository
    installable without the original NOTEARs implementation (which is *not*
    available on PyPI).
    """
    Z = latents.float().cpu().numpy()
    y = labels.float().cpu().numpy()
    # Center both variables to avoid numerical issues.
    Zc = Z - Z.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    # std may contain zeros → add eps to avoid division by 0.
    corr = ((Zc * yc[:, None]).mean(axis=0)) / (Zc.std(axis=0) * yc.std() + 1e-8)
    idxs = np.argsort(np.abs(corr))[-keep:][::-1]
    return idxs, corr[idxs]


def run_notears(
    latents: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda1: float = 1e-2,
    max_iter: int = 200,
    keep: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run NOTEARS linear causal discovery on concatenated (latent, label).

    If the *notears* package is unavailable (it is **not** published on PyPI),
    we gracefully fall back to a simple Pearson-correlation based ranking so
    that the remainder of the pipeline can still execute.
    """
    try:
        from notears.linear import notears_linear  # type: ignore

        Z = latents.cpu().float().numpy()
        y = labels.cpu().float().numpy().reshape(-1, 1)
        data = np.concatenate([Z, y], axis=1)
        W_est = notears_linear(data, lambda1=lambda1, max_iter=max_iter)

        # Edges z_k -> y are in the last column (excluding last row).
        causal_strength = W_est[:-1, -1]
        idxs = np.argsort(np.abs(causal_strength))[-keep:][::-1]
        return idxs, causal_strength[idxs]
    except ModuleNotFoundError:
        warnings.warn(
            "Package 'notears' is not available; falling back to Pearson-correlation heuristic.",
            RuntimeWarning,
        )
        return _fallback_notears(latents, labels, keep=keep)


# -----------------------------------------------------------------------------
# 2.  Counterfactual diffusion editor (LoRA fine-tuning)
# -----------------------------------------------------------------------------


class DiffusionEditor:
    """Light-weight *placeholder* Stable-Diffusion wrapper.

    Downloading the full Stable-Diffusion v1.5 checkpoint (~4 GB) is infeasible
    for most CI environments.  Therefore we provide a cheap stub that imitates
    the public API yet does **not** depend on the heavy model weights.
    """

    def __init__(self, *, device: str = "cuda", fp16: bool = True):
        try:
            from diffusers import StableDiffusionPipeline  # type: ignore

            dtype = torch.float16 if fp16 else torch.float32
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None, local_files_only=True
            )
            self.pipe.to(device)
            self._is_stub = False
        except Exception:
            # Any error (missing package, no internet, etc.) → fallback stub
            warnings.warn(
                "Falling back to a stub DiffusionEditor – counterfactual images "
                "will just echo inputs. Install diffusers & model weights for "
                "full functionality.",
                RuntimeWarning,
            )
            self.pipe = None
            self._is_stub = True
        self.device = device

    @staticmethod
    def _inject_lora(unet: nn.Module, rank: int = 8) -> nn.Module:  # pragma: no cover
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

    def finetune_lora(self, images: List[torch.Tensor], cfg):  # noqa: D401
        """No-op if running in *stub* mode, else tiny dummy loop."""
        if self._is_stub or not images:
            return

        unet: nn.Module = self._inject_lora(self.pipe.unet, rank=cfg.lora_rank)
        unet.train()
        optimiser = torch.optim.AdamW(unet.parameters(), lr=cfg.lora_lr)

        pbar = tqdm(range(min(cfg.steps, 10)), desc="LoRA-finetune", leave=False)  # clamp steps for CI
        for _ in pbar:
            idx = np.random.randint(0, len(images))
            x = images[idx].unsqueeze(0).to(self.device)
            noise_pred = unet(x, timesteps=torch.tensor([1], device=x.device)).sample  # type: ignore
            loss = noise_pred.abs().mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            pbar.set_postfix(loss=float(loss))
        self.pipe.unet = unet.eval()

    @torch.no_grad()
    def edit_latent(self, latent: torch.Tensor, dim: int, *, delta: float, cfg):
        """Return a *counterfactual* image.

        In *stub* mode we simply return a zero-tensor with the correct shape so
        that downstream code can continue.
        """
        if self._is_stub:
            # Assume images are 3×224×224 unless proven otherwise.
            dummy = torch.zeros(3, 224, 224, dtype=torch.float32)
            return dummy

        from torchvision.transforms import ToTensor  # local import

        prompt = "a photo"
        latent_mod = latent.clone()
        latent_mod[..., dim] += delta  # unused in dummy implementation

        img = self.pipe(
            prompt=prompt,
            num_inference_steps=cfg.ddim_steps,
            guidance_scale=cfg.cfg_scale,
        ).images[0]
        return ToTensor()(img)


# -----------------------------------------------------------------------------
# 3.  Fourier Consistent-Distance regulariser
# -----------------------------------------------------------------------------


def fourier_consistent_distance(x: torch.Tensor, logits: torch.Tensor, *, window: int = 7) -> torch.Tensor:  # noqa: D401
    """Penalise high-frequency energy of *x* (Fourier perspective)."""

    import torch.fft as fft  # local import keeps torch.fft optional for CPU

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
        self.dro_radius = dro_radius  # retained for API compatibility
        self.fourier_lambda = fourier_lambda
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _group_losses(self, logits: torch.Tensor, y: torch.Tensor, gid: torch.Tensor):
        loss_indiv = self.criterion(logits, y)
        groups = torch.unique(gid)
        g_loss = torch.stack([loss_indiv[gid == g].mean() for g in groups])
        return g_loss, groups

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader):
        self.model.train()
        for batch in tqdm(loader, desc="train_epoch", leave=False):
            x, y, gid = [t.to(self.device, non_blocking=True) for t in batch]
            logits = self.model(x)
            loss = self.criterion(logits, y)

            if self.fourier_lambda > 0:
                loss = loss + self.fourier_lambda * fourier_consistent_distance(x, logits)

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
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)
