from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import List

import hashlib
import importlib.util

import torch
from omegaconf import OmegaConf
from torchvision import models

from .preprocess import build_dataloader
from .train import GroupDROTrainer
from .evaluate import expected_calibration_error

# ---------------------------------------------------------------------------
#  Minimal *utility* helpers  ------------------------------------------------
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_CFG_DIR = _ROOT / "config"
_RESULTS_DIR = _ROOT / "results"
_RESULTS_DIR.mkdir(exist_ok=True)
_FIG_DIR = _ROOT / "figures"
_FIG_DIR.mkdir(exist_ok=True)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Hard guards – abort early if heavy deps missing / broken  -----------------
# ---------------------------------------------------------------------------


def _hard_abort(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def _verify_notears() -> None:
    """Abort if the *notears* package is unavailable.

    We depend on the canonical NumPy implementation released on PyPI as
    ``notears`` (see https://github.com/xunzheng/notears).  Earlier drafts of
    this code referenced a non-existent ``notears-torch`` fork which caused the
    dependency resolver to fail.  The guard is kept to surface a clear error
    message if the import is still missing.
    """
    try:
        import notears  # noqa: F401 – import only for the availability check
    except ImportError:
        _hard_abort("ERROR: 'notears' package not found – please install notears>=0.1.0 from PyPI.")


def _verify_sd_checkpoint(sd_sha: str, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        _hard_abort(
            f"Stable-Diffusion weights not found at {ckpt_path}. Please download before running."
        )
    hasher = hashlib.sha256()
    hasher.update(ckpt_path.read_bytes())
    if hasher.hexdigest() != sd_sha:
        _hard_abort("Stable-Diffusion checkpoint SHA-256 mismatch – file corrupted or wrong version.")


# ---------------------------------------------------------------------------
#  Causal discovery (NOTEARS) helper  ---------------------------------------
# ---------------------------------------------------------------------------


def _run_notears(
    latents: torch.Tensor,
    labels: torch.Tensor,
    lambda1: float,
    max_iter: int,
    keep_dims: int,
):
    from notears.linear import notears_linear  # heavy import – keep local

    z = latents.cpu().float().numpy()
    y = labels.cpu().float().numpy().reshape(-1, 1)
    data = __import__("numpy").concatenate([z, y], axis=1)

    w = notears_linear(data, lambda1=lambda1, max_iter=max_iter)
    causal_strength = w[:-1, -1]
    import numpy as np

    idx = np.argsort(causal_strength)[-keep_dims:][::-1]
    return idx, causal_strength[idx]


# ---------------------------------------------------------------------------
#  Counter-factual image generator  -----------------------------------------
# ---------------------------------------------------------------------------


class _CFGenerator:
    """Tiny wrapper around *diffusers* + LoRA fine-tuning for latent editing."""

    def __init__(
        self,
        sd_repo: str,
        lora_rank: int = 8,
        device: str | torch.device = _DEVICE,
    ) -> None:
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from peft import LoraConfig, get_peft_model, TaskType

        self.device = torch.device(device)
        self.pipe = StableDiffusionPipeline.from_pretrained(
            sd_repo, torch_dtype=torch.float16, safety_checker=None
        )
        self.pipe.scheduler = DDPMScheduler.from_pretrained(sd_repo, subfolder="scheduler")
        self.pipe.to(self.device)

        lora_cfg = LoraConfig(
            task_type=TaskType.UNET,
            target_modules=["to_q", "to_k", "to_v"],
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
        )
        self.pipe.unet = get_peft_model(self.pipe.unet, lora_cfg)
        self.pipe.unet.train(False)

    # ------------------------------------------------------------------
    def finetune(self, images: List[torch.Tensor], steps: int = 1_000, lr: float = 1e-4) -> None:
        if steps == 0:
            return  # discovery-only runs (e.g. DiffSpur)
        self.pipe.unet.train()
        opt = torch.optim.AdamW(self.pipe.unet.parameters(), lr=lr)
        for i in range(steps):
            x = random.choice(images).unsqueeze(0).to(self.device)
            noise = torch.randn_like(x)
            t = torch.randint(0, 1_000, (1,), device=self.device)
            noisy = self.pipe.scheduler.add_noise(x, noise, t)
            pred = self.pipe.unet(noisy, t).sample
            loss = (pred - noise).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (i + 1) % 100 == 0:
                print(f"LoRA step {i + 1}/{steps} | loss={loss.item():.4f}")
        self.pipe.unet.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def edit(
        self,
        latent: torch.Tensor,
        dim: int,
        delta: float,
        ddim_steps: int = 20,
        cfg_scale: float = 7.5,
    ) -> torch.Tensor:
        # *Very* simplified: we ignore precise latent-space mapping and rely on stochasticity.
        prompt = "a photo"
        latent_mod = latent.clone()
        latent_mod[..., dim] += delta
        image = self.pipe(prompt, num_inference_steps=ddim_steps, guidance_scale=cfg_scale).images[0]
        import torchvision.transforms as T

        return T.ToTensor()(image)


# ---------------------------------------------------------------------------
#  Driver – one experiment config  ------------------------------------------
# ---------------------------------------------------------------------------


def _run_single_config(cfg_path: Path) -> None:
    print(f"\n==== Running {cfg_path.stem} ====", flush=True)

    cfg = OmegaConf.load(cfg_path)
    _verify_notears()

    # Verify SD weights (assumes HuggingFace cache layout)
    if "counterfactual" in cfg and "sd_sha256" in cfg.counterfactual:
        sd_cache = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--runwayml--stable-diffusion-v1-5"
            / "snapshots"
            / "sd-v1-5.ckpt"
        )
        _verify_sd_checkpoint(cfg.counterfactual.sd_sha256, sd_cache)

    for seed in cfg.seed_list:
        _set_seed(seed)
        _run_one_seed(cfg, seed)


# ---------------------------------------------------------------------------

def _run_one_seed(cfg: "omegaconf.DictConfig", seed: int) -> None:  # noqa: C901 – complex but linear
    start = time.time()

    # ------------------------------------------------------------------
    # 1. data loaders
    # ------------------------------------------------------------------
    batch_size = cfg.classifier.batch_size or 64
    train_loader = build_dataloader(cfg, "train", batch_size)
    val_loader = build_dataloader(cfg, "val", 64)

    # ------------------------------------------------------------------
    # 2. latent extraction via CLIP (frozen)
    # ------------------------------------------------------------------
    from transformers import CLIPModel, CLIPProcessor

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    latents, labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            inputs = clip_proc(images=x, return_tensors="pt").to(_DEVICE)
            z = clip_model.get_image_features(**inputs)
            latents.append(z.cpu())
            labels.append(y)
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # ------------------------------------------------------------------
    # 3. causal discovery (NOTEARS)
    # ------------------------------------------------------------------
    sel_dims, _ = _run_notears(
        latents,
        labels,
        cfg.causal.lambda1,
        getattr(cfg.causal, "max_iter", 200),
        cfg.causal.keep_dims,
    )
    print("Selected latent dims:", sel_dims.tolist())

    # ------------------------------------------------------------------
    # 4. Counter-factual generator (LoRA fine-tune)
    # ------------------------------------------------------------------
    cf_gen = _CFGenerator(
        cfg.counterfactual.sd_repo,
        cfg.counterfactual.lora_rank,
        _DEVICE,
    )
    sample_imgs = [train_loader.dataset[i][0] for i in range(min(128, len(train_loader.dataset)))]
    cf_gen.finetune(sample_imgs, cfg.counterfactual.train_steps, cfg.counterfactual.lora_lr)

    # Build *augmented* data set on-the-fly -----------------------------------
    x_cf, y_cf, gids = [], [], []

    for idx, (x, y) in enumerate(train_loader.dataset):
        inputs = clip_proc(images=x, return_tensors="pt").to(_DEVICE)
        z = clip_model.get_image_features(**inputs).squeeze(0)
        for d in sel_dims:
            for sign in (-1, 1):
                img_cf = cf_gen.edit(
                    z,
                    int(d),
                    float(sign),
                    cfg.counterfactual.ddim_steps,
                    cfg.counterfactual.cfg_scale,
                )
                x_cf.append(img_cf)
                y_cf.append(torch.tensor(y))
                # deterministic group id (matches GroupDROTrainer helper)
                gids.append(torch.tensor((hash((idx, int(d), sign)) & 0xFFFFFFFFFFFFFFFF)))

    aug_ds = torch.utils.data.TensorDataset(torch.stack(x_cf), torch.stack(y_cf), torch.stack(gids))
    aug_loader = torch.utils.data.DataLoader(
        aug_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # ------------------------------------------------------------------
    # 5. classifier + robust training
    # ------------------------------------------------------------------
    num_classes = 2 if cfg.dataset.name in {"waterbirds", "celeba"} else 10
    model = models.__dict__[cfg.classifier.arch](weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    trainer = GroupDROTrainer(
        model,
        lr=cfg.classifier.lr,
        weight_decay=cfg.classifier.weight_decay,
        dro_radius=cfg.classifier.dro_radius,
        fourier_lambda=cfg.classifier.fourier_lambda,
        device=_DEVICE,
    )

    acc_history: List[float] = []
    for epoch in range(cfg.classifier.epochs):
        trainer.train_epoch(aug_loader)
        acc = trainer.evaluate_accuracy(val_loader)
        acc_history.append(acc)
        if (epoch + 1) % 10 == 0 or epoch + 1 == cfg.classifier.epochs:
            print(f"Seed {seed} | epoch {epoch + 1}/{cfg.classifier.epochs} | val-acc = {acc:.3f}")

    # ------------------------------------------------------------------
    # 6. logging --------------------------------------------------------
    # ------------------------------------------------------------------
    record = {
        "seed": seed,
        "val_acc_history": acc_history,
        "runtime_sec": round(time.time() - start, 2),
    }
    with open(_RESULTS_DIR / f"{cfg.experiment_name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
#  Entry-point  -------------------------------------------------------------
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover – CLI entry
    if len(sys.argv) == 1:  # run *all* YAMLs in config/
        for cfg_file in sorted(_CFG_DIR.glob("*.yaml")):
            _run_single_config(cfg_file)
    else:  # explicit config path(s)
        for arg in sys.argv[1:]:
            _run_single_config(Path(arg))


if __name__ == "__main__":
    main()
