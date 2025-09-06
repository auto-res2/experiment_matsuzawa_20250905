"""
main.py – orchestrates the three experiments using the refactored helper
modules.  It is intended to be run via

    python -m src.main
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, Dataset

import yaml
from tqdm import tqdm

from .preprocess import (
    set_seed,
    download_hf_dataset,
    transforms_224,
)
from .train import ConceptMiner, CausalGenerator, GCDROLoss
from .evaluate import ExperimentBase

# ---------------------------------------------------------------------------
#                           CONFIG  LOADING
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("config/config.yaml")
if not CONFIG_PATH.exists():
    raise RuntimeError("config/config.yaml not found – project improperly initialised.")
with open(CONFIG_PATH) as fp:
    CONFIG: Dict[str, Any] = yaml.safe_load(fp)

# ensure result & data directories exist ------------------------------------
Path(CONFIG["global"]["results_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["global"]["data_root"]).mkdir(parents=True, exist_ok=True)

DEVICE = CONFIG["global"]["device"]

# ---------------------------------------------------------------------------
#                          EXPERIMENT  1 –  WATERBIRDS
# ---------------------------------------------------------------------------
class Exp1_Waterbirds(ExperimentBase):
    def __init__(self):
        super().__init__(
            "exp1_WATERBIRDS", CONFIG["experiments"]["exp1"], CONFIG["global"]
        )

    # ------------------------------------------------------------------
    def run(self):
        wb_root = download_hf_dataset(
            CONFIG["datasets"]["waterbirds"]["hf_repo"], data_root=CONFIG["global"]["data_root"]
        )
        split_cfg = CONFIG["datasets"]["waterbirds"]["split"]
        train_ds = tv.datasets.ImageFolder(
            wb_root / split_cfg["train"], transform=transforms_224(train=True)
        )
        val_ds = tv.datasets.ImageFolder(
            wb_root / split_cfg["val"], transform=transforms_224(train=False)
        )
        test_ds = tv.datasets.ImageFolder(
            wb_root / split_cfg["test"], transform=transforms_224(train=False)
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            num_workers=CONFIG["global"]["num_workers"],
        )

        for seed in CONFIG["global"]["seeds"]:
            set_seed(seed)
            t0 = time.time()

            # 1) concept mining ------------------------------------------------
            miner = ConceptMiner(
                device=DEVICE, num_clusters=self.cfg["concept_k"]
            ).fit(train_dl, max_images=2000)
            labels_np = np.array([y for _, y in train_ds])
            order = miner.rank_by_mutual_information(train_dl, labels_np)
            _ = order[:20]  # selected spurious concepts – not used further here

            # 2) counterfactual generation -----------------------------------
            gen = CausalGenerator(DEVICE, CONFIG["models"]["sdxl_base"])
            from PIL import Image  # local import – pillow is a hard dependency

            cf_imgs, cf_lbls, cf_gvec = [], [], []
            cf_dir = self.results_dir / f"cf_seed{seed}"; cf_dir.mkdir(exist_ok=True)
            for idx, (img, lbl) in enumerate(train_ds):
                if np.random.rand() > self.cfg["cf_ratio"]:
                    continue
                mask = Image.new("L", img.size, 0)
                mask.paste(255, [int(0.2 * img.width), int(0.2 * img.height), int(0.8 * img.width), int(0.8 * img.height)])
                try:
                    cf_img = gen.generate_cf(img.convert("RGB"), mask)
                except Exception as e:
                    raise RuntimeError("Counterfactual generation failed") from e
                cf_img.save(cf_dir / f"cf_{idx}.png")
                cf_imgs.append(transforms_224(train=True)(cf_img))
                cf_lbls.append(lbl)
                cf_gvec.append(miner.concept_presence(transforms_224(train=False)(img)))

            if not cf_imgs:
                raise RuntimeError("No counterfactuals produced – aborting experiment.")
            cf_tensor = torch.stack(cf_imgs)
            cf_labels = torch.tensor(cf_lbls)
            cf_gvec = torch.tensor(np.stack(cf_gvec))

            # 3) classifier training -----------------------------------------
            import timm

            model = timm.create_model(
                "vit_small_patch16_224", pretrained=True, num_classes=len(train_ds.classes)
            ).to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), **CONFIG["global"]["optim"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg["epochs"]
            )
            dro_loss = GCDROLoss(alpha=self.cfg["lambda_gc"])

            metrics = {"train_loss": [], "val_acc": []}
            for ep in range(self.cfg["epochs"]):
                model.train()
                batch_losses = []
                for imgs, lbls in tqdm(train_dl, desc=f"[Train] epoch {ep}"):
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    loss = F.cross_entropy(model(imgs), lbls)
                    loss.backward(); optimizer.step(); optimizer.zero_grad()
                    batch_losses.append(loss.item())

                # Counterfactual pass -----------------------------------
                idx = torch.randperm(cf_tensor.size(0))[: CONFIG["global"]["batch_size"]["exp1"]]
                out_cf = model(cf_tensor[idx].to(DEVICE))
                loss_cf = (
                    F.mse_loss(out_cf.softmax(-1), out_cf.softmax(-1)) * self.cfg["lambda_cc"]
                    + dro_loss(out_cf, cf_labels[idx].to(DEVICE), cf_gvec[idx].to(DEVICE))
                )
                loss_cf.backward(); optimizer.step(); optimizer.zero_grad()

                scheduler.step()
                metrics["train_loss"].append(float(np.mean(batch_losses)))

                # validation accuracy ----------------------------------
                model.eval(); correct = total = 0
                with torch.no_grad():
                    for imgs, lbls in DataLoader(val_ds, batch_size=128, num_workers=4):
                        preds = model(imgs.to(DEVICE)).argmax(1)
                        correct += (preds.cpu() == lbls).sum().item(); total += lbls.size(0)
                val_acc = 100.0 * correct / total
                metrics["val_acc"].append(val_acc)
                print(f"epoch {ep} – valAcc: {val_acc:.2f}")

            # 4) test evaluation ----------------------------------------------
            model.eval(); correct = total = 0
            with torch.no_grad():
                for imgs, lbls in DataLoader(test_ds, batch_size=128, num_workers=4):
                    preds = model(imgs.to(DEVICE)).argmax(1)
                    correct += (preds.cpu() == lbls).sum().item(); total += lbls.size(0)
            test_acc = 100.0 * correct / total

            wall = time.time() - t0
            result = {
                "seed": seed,
                "test_accuracy": test_acc,
                "val_accuracy_last": metrics["val_acc"][-1],
                "train_loss_last": metrics["train_loss"][-1],
                "epoch_cnt": self.cfg["epochs"],
                "wall_clock_sec": wall,
                "gpu_util_percent": self._gpu_util(),
            }
            self.log_and_save(seed, result)
            # figure ----------------------------------------------------
            self.save_line_plot(metrics["val_acc"], list(range(self.cfg["epochs"])), "Val Accuracy [%]", "accuracy_curve.pdf")


# ---------------------------------------------------------------------------
#                     EXPERIMENT 2  –  CELEBA  (TINT SETUP)
# ---------------------------------------------------------------------------
class Exp2_CelebA(ExperimentBase):
    def __init__(self):
        super().__init__("exp2_CELEBA", CONFIG["experiments"]["exp2"], CONFIG["global"])

    # ------------------------------------------------------------------
    def run(self):
        celeba_root = download_hf_dataset(
            CONFIG["datasets"]["celeba"]["hf_repo"], data_root=CONFIG["global"]["data_root"]
        )
        img_dir = celeba_root / "img_align_celeba_png"
        attr_file = celeba_root / "list_attr_celeba.txt"
        if not img_dir.exists():
            raise RuntimeError("CelebA images not found after download.")

        # Parse attributes --------------------------------------------
        with attr_file.open() as fp:
            lines = fp.read().splitlines()[2:]
        img_to_attrs = {ln.split()[0]: [int(x) for x in ln.split()[1:]] for ln in lines}
        idx_blond, idx_male = 9, 20  # indices in attribute list (0-based)

        from PIL import Image

        class CelebATint(Dataset):
            def __init__(self, split: str, train_tint: bool):
                self.paths = sorted(img_dir.glob("*.png"))
                n_train = int(0.9 * len(self.paths))
                self.paths = self.paths[:n_train] if split == "train" else self.paths[n_train:]
                self.train_tint = train_tint
                self.tfms = transforms_224(train=(split == "train"))

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                p = self.paths[idx]
                img = Image.open(p).convert("RGB")
                attrs = img_to_attrs[p.name]
                y = 1 if attrs[idx_blond] == 1 else 0
                male = 1 if attrs[idx_male] == 1 else 0
                if self.train_tint:
                    tint = (255, 0, 0) if male == 0 else (0, 0, 255)
                    for x in range(10):
                        for y_pix in range(10):
                            img.putpixel((x, y_pix), tint)
                return self.tfms(img), y

        train_ds = CelebATint("train", train_tint=True)
        test_ds = CelebATint("test", train_tint=False)
        train_dl = DataLoader(
            train_ds,
            batch_size=CONFIG["global"]["batch_size"]["exp2"],
            shuffle=True,
            num_workers=CONFIG["global"]["num_workers"],
        )

        import timm
        for seed in CONFIG["global"]["seeds"]:
            set_seed(seed)
            model = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=2).to(DEVICE)
            optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
            # -------- ERM pre-training --------------------------------
            for ep in range(self.cfg["epochs_erm"]):
                model.train()
                for imgs, lbls in tqdm(train_dl, desc=f"[ERM] epoch {ep}"):
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    loss = F.cross_entropy(model(imgs), lbls)
                    loss.backward(); optim.step(); optim.zero_grad()

            # accuracy before fine-tuning ------------------------------
            def _acc(ds):
                correct = total = 0
                with torch.no_grad():
                    for imgs, lbls in DataLoader(ds, batch_size=128):
                        preds = model(imgs.to(DEVICE)).argmax(1).cpu()
                        correct += (preds == lbls).sum().item(); total += lbls.size(0)
                return 100.0 * correct / total

            acc_before = _acc(test_ds)

            # concept mining (steps 1-3 simplified) --------------------
            miner = ConceptMiner(device=DEVICE, num_clusters=self.cfg["concept_k"]).fit(train_dl, max_images=1000)
            labels_np = np.array([y for _, y in train_ds])
            top_concept = int(miner.rank_by_mutual_information(train_dl, labels_np)[0])

            # fine-tune with counterfactual consistency ---------------
            for ep in range(self.cfg["epochs_finetune"]):
                model.train()
                for imgs, lbls in tqdm(train_dl, desc=f"[FT] epoch {ep}"):
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    logits = model(imgs)
                    imgs_cf = imgs.clone(); imgs_cf[:, :, :10, :10] = 1 - imgs_cf[:, :, :10, :10]
                    logits_cf = model(imgs_cf)
                    loss = F.cross_entropy(logits, lbls) + F.mse_loss(
                        logits.softmax(-1), logits_cf.softmax(-1)
                    )
                    loss.backward(); optim.step(); optim.zero_grad()

            acc_after = _acc(test_ds)
            self.log_and_save(seed, {
                "seed": seed,
                "acc_before": acc_before,
                "acc_after": acc_after,
                "acc_gain": acc_after - acc_before,
                "top_concept_id": top_concept,
            })
            self.save_line_plot([acc_before, acc_after], [0, 1], "Accuracy [%]", "accuracy_pair1.pdf")


# ---------------------------------------------------------------------------
#                     EXPERIMENT 3  –  NICO DATASET
# ---------------------------------------------------------------------------
class Exp3_NICO(ExperimentBase):
    def __init__(self):
        super().__init__("exp3_NICO", CONFIG["experiments"]["exp3"], CONFIG["global"])

    # ------------------------------------------------------------------
    def run(self):
        nico_zip = Path(CONFIG["global"]["data_root"]) / "NICO.zip"
        if not nico_zip.exists():
            import requests
            from tqdm import tqdm as tq

            print("[Info ] Downloading NICO dataset (~1.6 GB)…")
            with requests.get(CONFIG["datasets"]["nico"]["zenodo_url"], stream=True) as r:
                r.raise_for_status()
                with open(nico_zip, "wb") as f:
                    for chunk in tq(r.iter_content(chunk_size=8192)):
                        if chunk:
                            f.write(chunk)
        nico_root = Path(CONFIG["global"]["data_root"]) / "NICO"
        if not nico_root.exists():
            print("[Info ] Extracting NICO …")
            shutil.unpack_archive(nico_zip, nico_root)

        train_ds = tv.datasets.ImageFolder(nico_root / "train", transform=transforms_224(train=True))
        test_ds = tv.datasets.ImageFolder(nico_root / "test", transform=transforms_224(train=False))
        train_dl = DataLoader(
            train_ds,
            batch_size=CONFIG["global"]["batch_size"]["exp3"],
            shuffle=True,
            num_workers=CONFIG["global"]["num_workers"],
        )

        import timm
        for seed in CONFIG["global"]["seeds"]:
            set_seed(seed)
            model = timm.create_model(
                "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=len(train_ds.classes)
            ).to(DEVICE)
            optim = torch.optim.AdamW(model.parameters(), **CONFIG["global"]["optim"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.cfg["epochs"])

            for ep in range(self.cfg["epochs"]):
                model.train()
                for imgs, lbls in tqdm(train_dl, desc=f"[NICO] epoch {ep}"):
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    F.cross_entropy(model(imgs), lbls).backward(); optim.step(); optim.zero_grad()
                sched.step()

            # evaluation ------------------------------------------------
            model.eval(); correct = total = 0
            with torch.no_grad():
                for imgs, lbls in DataLoader(test_ds, batch_size=128):
                    preds = model(imgs.to(DEVICE)).argmax(1).cpu()
                    correct += (preds == lbls).sum().item(); total += lbls.size(0)
            acc = 100.0 * correct / total
            self.log_and_save(seed, {"seed": seed, "NICO_test_acc": acc})


# ---------------------------------------------------------------------------
#                                    MAIN
# ---------------------------------------------------------------------------

def main():
    Exp1_Waterbirds().run()
    Exp2_CelebA().run()
    Exp3_NICO().run()


if __name__ == "__main__":
    main()
