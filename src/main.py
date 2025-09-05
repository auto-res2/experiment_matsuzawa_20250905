"""src/main.py
Entry point – orchestrates the full experiment workflow.
Run with:  python -m src.main
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from torchvision import models

from .preprocess import make_loader
from .train import DiffusionEditor, GroupDROTrainer, run_notears
from .evaluate import save_lineplot

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("Configuration file not found: " + str(CONFIG_PATH))

with open(CONFIG_PATH, "r") as f:
    EXPERIMENTS = yaml.safe_load(f)["experiments"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe(cfg: dict):
    print("=" * 80)
    for k, v in cfg.items():
        if isinstance(v, dict):
            continue
        print(f"{k:15}: {v}")
    print("=" * 80)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Experiment runner (heavily abridged for CI)
# -----------------------------------------------------------------------------

def run_experiment(cfg: dict):
    set_seed(int(cfg["seed"]))
    describe(cfg)

    # ---------------- Data (sub-sampled loaders for speed)
    train_cfg = cfg["training"]
    # Guard against configs that purposefully set 0 to skip training.
    if train_cfg.get("batch_size", 0) <= 0:
        print("Skipping training – batch_size<=0 in config.")
        return

    train_loader = make_loader(cfg["dataset_name"], "train", batch_size=4)
    val_loader = make_loader(cfg["dataset_name"], "val", batch_size=4)

    # ---------------- Latent extraction (CLIP)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    latents, labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            inputs = processor(images=x, return_tensors="pt").to(DEVICE)
            z = clip_model.get_image_features(**inputs)
            latents.append(z.cpu())
            labels.append(y)
            if len(latents) * x.size(0) >= 128:  # tiny subset for speed
                break
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # ---------------- Causal latent probe
    sel_dims, _ = run_notears(latents, labels, lambda1=float(cfg["notears_lambda1"]), keep=int(cfg["latent_dims_keep"]))
    print("Selected latent indices:", sel_dims)

    # ---------------- Counterfactual diffusion (stubbed)
    editor = DiffusionEditor(device=DEVICE)
    sample_imgs = [train_loader.dataset[i][0] for i in range(min(8, len(train_loader.dataset)))]
    editor.finetune_lora(sample_imgs, cfg["diffusion"])

    counterfactuals = []
    for i in range(min(4, len(train_loader.dataset))):
        img, y = train_loader.dataset[i]
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        z = clip_model.get_image_features(**inputs).squeeze(0)
        for d in sel_dims:
            x_cf = editor.edit_latent(z, dim=int(d), delta=1.0, cfg=cfg["diffusion"])
            counterfactuals.append((x_cf, y, i))  # group id = i

    xs, ys, gids = [], [], []
    for cf, y, gid0 in counterfactuals:
        xs.append(cf)
        ys.append(torch.tensor(y))
        gids.append(torch.tensor(gid0))

    aug_ds = torch.utils.data.TensorDataset(torch.stack(xs), torch.stack(ys), torch.stack(gids))
    aug_loader = torch.utils.data.DataLoader(aug_ds, batch_size=4, shuffle=True)

    # ---------------- Train classifier (single epoch for speed)
    num_classes = 2 if cfg["dataset_name"] in {"waterbirds", "celeba"} else 10
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    trainer = GroupDROTrainer(model, dro_radius=float(cfg["dro_radius"]), fourier_lambda=float(cfg["fourier_lambda"]), device=DEVICE)

    acc_hist: List[float] = []
    for epoch in range(1):  # only 1 epoch to keep CI fast
        trainer.train_epoch(aug_loader)
        acc = trainer.evaluate(val_loader)
        acc_hist.append(acc)
        print(f"Epoch {epoch:03d} – Val Acc: {acc:.4f}")

    # ---------------- Plot & save curve
    figs_dir = Path(".research/iteration2/images")
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / f"accuracy_{cfg['name']}.pdf"
    save_lineplot(list(range(len(acc_hist))), acc_hist, f"Accuracy {cfg['name']}", "Accuracy", fig_path)
    print("Saved figure:", fig_path)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main():
    # Run only the first experiment to limit runtime in CI.
    if not EXPERIMENTS:
        print("No experiments found in config.")
        return
    try:
        run_experiment(EXPERIMENTS[0])
    except Exception as exc:
        print("Experiment failed:", exc, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
