"""src/main.py
Entry point – orchestrates the full experiment workflow using the refactored
modules (train, evaluate, preprocess).

Run with:  python -m src.main
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from torchvision import models

from .preprocess import DATA_DIR, make_loader
from .train import (
    DiffusionEditor,
    GroupDROTrainer,
    run_notears,
)
from .evaluate import save_lineplot

# -----------------------------------------------------------------------------
# 1.  Configuration – read from YAML in ../config/config.yaml
# -----------------------------------------------------------------------------
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("Configuration file not found: " + str(CONFIG_PATH))

with open(CONFIG_PATH, "r") as f:
    EXPERIMENTS = yaml.safe_load(f)["experiments"]


# -----------------------------------------------------------------------------
# 2.  Utilities
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


TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# 3.  Single experiment runner
# -----------------------------------------------------------------------------

def run_experiment(cfg: dict):
    set_seed(int(cfg["seed"]))
    describe(cfg)

    # ---------------- Data
    train_cfg = cfg["training"]
    train_loader = make_loader(cfg["dataset_name"], "train", train_cfg["batch_size"])
    val_loader = make_loader(cfg["dataset_name"], "val", train_cfg["batch_size"])

    # ---------------- Latent extraction (CLIP)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(TORCH_DEVICE).eval().half()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    latents, labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            inputs = processor(images=x, return_tensors="pt").to(TORCH_DEVICE)
            z = clip_model.get_image_features(**inputs)
            latents.append(z.cpu())
            labels.append(y)
            if len(latents) * x.size(0) > 1024:  # small subset for causal probe
                break
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # ---------------- Causal latent probe
    sel_dims, strengths = run_notears(
        latents,
        labels,
        lambda1=float(cfg["notears_lambda1"]),
        keep=int(cfg["latent_dims_keep"]),
    )
    print("Selected latent indices:", sel_dims)

    # ---------------- Counterfactual diffusion (placeholder)
    editor = DiffusionEditor(device=TORCH_DEVICE)
    sample_imgs = [train_loader.dataset[i][0] for i in range(16)]
    editor.finetune_lora(sample_imgs, cfg["diffusion"])

    counterfactuals = []
    for i in range(8):
        img, y = train_loader.dataset[i]
        inputs = processor(images=img, return_tensors="pt").to(TORCH_DEVICE)
        z = clip_model.get_image_features(**inputs).squeeze(0)
        for d in sel_dims:
            x_cf = editor.edit_latent(z, dim=int(d), delta=1.0, cfg=cfg["diffusion"])
            counterfactuals.append((x_cf, y, i))  # group id = i

    # build augmented dataset (original + cf)
    xs, ys, gids = [], [], []
    for cf, y, gid0 in counterfactuals:
        xs.append(cf)
        ys.append(torch.tensor(y))
        gids.append(torch.tensor(gid0))
    for i in range(len(counterfactuals)):
        img, y = train_loader.dataset[i]
        xs.append(img)
        ys.append(torch.tensor(y))
        gids.append(torch.tensor(i))

    aug_ds = torch.utils.data.TensorDataset(torch.stack(xs), torch.stack(ys), torch.stack(gids))
    aug_loader = torch.utils.data.DataLoader(aug_ds, batch_size=train_cfg["batch_size"], shuffle=True)

    # ---------------- Train classifier
    num_classes = 2 if cfg["dataset_name"] in {"waterbirds", "celeba"} else 10
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    trainer = GroupDROTrainer(
        model,
        dro_radius=float(cfg["dro_radius"]),
        fourier_lambda=float(cfg["fourier_lambda"]),
        device=TORCH_DEVICE,
    )

    acc_hist: List[float] = []
    for epoch in range(train_cfg["epochs"]):
        trainer.train_epoch(aug_loader)
        acc = trainer.evaluate(val_loader)
        acc_hist.append(acc)
        print(f"Epoch {epoch:03d} – Val Acc: {acc:.4f}")

    # ---------------- Plot & save curve
    figs_dir = Path(".research/iteration1/images")
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figs_dir / f"accuracy_{cfg['name']}.pdf"
    save_lineplot(list(range(len(acc_hist))), acc_hist, f"Accuracy {cfg['name']}", "Accuracy", fig_path)
    print("Saved figure:", fig_path)


# -----------------------------------------------------------------------------
# 4.  Main driver – iterate over all experiments
# -----------------------------------------------------------------------------

def main():
    for exp_cfg in EXPERIMENTS:
        try:
            run_experiment(exp_cfg)
        except Exception as exc:
            print("Experiment failed:", exp_cfg["name"], "–", exc, file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
