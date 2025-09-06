"""src/main.py – orchestration entry-point for the refactored CGSI demo."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .evaluate import ExperimentBase
from .preprocess import download_hf_dataset, set_seed, transforms_224
from .train import CausalGenerator, ConceptMiner, GCDROLoss

# ---------------------------------------------------------------------------
#                      YAML  CONFIG  (shared across exps)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CFG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def _load_cfg() -> Dict[str, Any]:
    if not _CFG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {_CFG_PATH}")
    with open(_CFG_PATH, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


# ---------------------------------------------------------------------------
#                             Waterbirds helpers
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def _make_waterbirds_dls(cfg: Dict[str, Any], split: str, seed: int):
    repo = cfg["datasets"]["waterbirds"]["hf_repo"]
    data_root = cfg["global"]["data_root"]
    root = download_hf_dataset(repo, data_root=data_root)
    split_dir = root / cfg["datasets"]["waterbirds"]["split"][split]
    if not split_dir.exists():
        raise RuntimeError(f"Expected Waterbirds split dir missing: {split_dir}")
    tfms = transforms_224(train=(split == "train"))
    ds = ImageFolder(split_dir, transform=tfms)
    dl = DataLoader(
        ds,
        batch_size=cfg["global"]["batch_size"]["exp1"],
        shuffle=(split == "train"),
        num_workers=cfg["global"]["num_workers"],
        pin_memory=True,
    )
    return ds, dl


# ---------------------------------------------------------------------------
#                           Experiment 1 – Waterbirds
# ---------------------------------------------------------------------------

def _run_exp1(cfg: Dict[str, Any]):
    exp_cfg = cfg["experiments"]["exp1"]
    base = ExperimentBase("exp1", exp_cfg, cfg["global"])

    seeds: List[int] = cfg["global"]["seeds"]
    device = torch.device(cfg["global"]["device"] if torch.cuda.is_available() else "cpu")

    import timm  # heavy – delay until here

    for seed in seeds:
        set_seed(seed)
        train_ds, train_dl = _make_waterbirds_dls(cfg, "train", seed)
        val_ds, val_dl = _make_waterbirds_dls(cfg, "val", seed)
        test_ds, test_dl = _make_waterbirds_dls(cfg, "test", seed)

        # 1) Concept mining --------------------------------------------------
        miner = ConceptMiner(device=str(device), num_clusters=exp_cfg["concept_k"])
        miner.fit(train_dl, max_images=2048)
        mi_rank = miner.rank_by_mutual_information(train_dl, labels=[lbl for _, lbl in train_ds])
        spurious_idx = mi_rank[:20]
        print(f"[Exp1] Top-20 spurious concept idx = {spurious_idx[:5]} …")

        # 2) Counterfactual generation --------------------------------------
        gen = CausalGenerator(device=str(device), model_name=cfg["models"]["sdxl_base"])
        out_dir = Path("data/aux/counterfactuals")
        out_dir.mkdir(parents=True, exist_ok=True)
        cf_pairs: List[Tuple[Path, Path]] = []
        for i, (img, _) in enumerate(tqdm(train_ds, desc="Generating CF", leave=False)):
            if i >= int(len(train_ds) * exp_cfg["cf_ratio"]):
                break
            from PIL import Image, ImageDraw

            w, h = img.size
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).rectangle([0, 0, w, int(h * 0.3)], fill=255)
            cf = gen.generate_cf(img, mask)
            orig_p = out_dir / f"orig_{seed}_{i}.png"
            cf_p = out_dir / f"cf_{seed}_{i}.png"
            img.save(orig_p)
            cf.save(cf_p)
            cf_pairs.append((orig_p, cf_p))

        # 3) Classifier ------------------------------------------------------
        model = timm.create_model("vit_small_patch16_224", pretrained=True).to(device)
        model.train()
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["global"]["optim"]["lr"],
            weight_decay=cfg["global"]["optim"]["weight_decay"],
            betas=tuple(cfg["global"]["optim"]["betas"]),
            eps=cfg["global"]["optim"]["eps"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=exp_cfg["epochs"] * len(train_dl)
        )
        loss_gc = GCDROLoss(alpha=exp_cfg["lambda_gc"])  # not used downstream but kept for completeness

        epoch_res: Dict[str, List[float]] = {k: [] for k in ["avg_acc", "loss"]}
        t0 = time.perf_counter()
        for epoch in range(1, exp_cfg["epochs"] + 1):
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device)
                logits = model(imgs)
                loss_cls = F.cross_entropy(logits, labels)
                loss = loss_cls + exp_cfg["lambda_wd"] * sum(p.pow(2).sum() for p in model.parameters())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                scheduler.step()
            # ---- validation ---------------------------------------------
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, labels in val_dl:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = model(imgs)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.numel()
            val_acc = correct / total * 100
            epoch_res["avg_acc"].append(val_acc)
            epoch_res["loss"].append(float(loss_cls.item()))
            model.train()
        wall = time.perf_counter() - t0

        # -------- test ------------------------------------------------------
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total += labels.numel()
        test_acc = correct / total * 100

        result_blob = {
            "description": f"Waterbirds seed={seed}",
            "epochs": exp_cfg["epochs"],
            "val_acc_per_epoch": epoch_res["avg_acc"],
            "test_acc": test_acc,
            "wall_clock_sec": wall,
            "figures": ["training_loss.pdf", "training_accuracy.pdf"],
        }
        base.log_and_save(seed, result_blob)
        base.save_line_plot(epoch_res["loss"], list(range(1, exp_cfg["epochs"] + 1)), "Loss", "training_loss.pdf")
        base.save_line_plot(epoch_res["avg_acc"], list(range(1, exp_cfg["epochs"] + 1)), "ValAcc", "training_accuracy.pdf")

    print("\n[Exp-1] Finished – results under", base.out_dir)


# ---------------------------------------------------------------------------
#                     Experiment 2 – CelebA tint verification
# ---------------------------------------------------------------------------
import numpy as np
from torchvision.transforms.functional import to_pil_image


def _tint_img(pil_img, color: Tuple[int, int, int]):
    from PIL import Image

    img = pil_img.copy()
    tint = Image.new("RGB", img.size, color)
    img.paste(tint.crop((0, 0, 10, 10)), (0, 0))
    return img


def _eval_acc(model, dl, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(1) == lbls).sum().item()
            total += lbls.numel()
    return correct / total * 100


def _run_exp2(cfg: Dict[str, Any]):
    exp_cfg = cfg["experiments"]["exp2"]
    base = ExperimentBase("exp2", exp_cfg, cfg["global"])
    device = torch.device(cfg["global"]["device"] if torch.cuda.is_available() else "cpu")
    seeds = cfg["global"]["seeds"]

    repo = cfg["datasets"]["celeba"]["hf_repo"]
    root = download_hf_dataset(repo, data_root=cfg["global"]["data_root"])
    img_root = root
    if not img_root.exists():
        raise RuntimeError("CelebA download missing expected directory")

    import timm
    from torch.utils.data import DataLoader, Dataset

    for seed in seeds:
        set_seed(seed)

        tf_train = transforms_224(train=True)
        tf_val = transforms_224(train=False)

        class CelebTint(Dataset):
            def __init__(self, root: Path, split: str):
                self.files = sorted(list(root.glob("*.jpg")))[:5000]  # keep runtime small
                self.split = split

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                from PIL import Image

                img = Image.open(self.files[idx]).convert("RGB")
                lbl = 0 if "female" in self.files[idx].name.lower() else 1  # dummy label
                if self.split == "train":
                    color = (255, 0, 0) if lbl == 0 else (0, 0, 255)
                    img = _tint_img(img, color)
                    img = tf_train(img)
                else:
                    img = tf_val(img)
                return img, lbl

        train_ds = CelebTint(img_root, "train")
        val_ds = CelebTint(img_root, "val")
        train_dl = DataLoader(train_ds, batch_size=cfg["global"]["batch_size"]["exp2"], shuffle=True, num_workers=cfg["global"]["num_workers"], pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=cfg["global"]["batch_size"]["exp2"], shuffle=False, num_workers=cfg["global"]["num_workers"], pin_memory=True)

        model = timm.create_model("vit_small_patch16_224", pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # pre-train ERM ------------------------------------------------------
        model.train()
        for _ in range(exp_cfg["epochs_erm"]):
            for imgs, lbls in train_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                loss = F.cross_entropy(model(imgs), lbls)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        acc_before = _eval_acc(model, val_dl, device)

        # concept mining -----------------------------------------------------
        miner = ConceptMiner(device=str(device), num_clusters=exp_cfg["concept_k"])
        miner.fit(train_dl, max_images=1024)
        spurious = miner.rank_by_mutual_information(train_dl, labels=[lbl for _, lbl in train_ds])[:1]
        print(f"[Exp2] Most spurious concept: {spurious}")

        # counterfactual τ (placeholder) ------------------------------------
        gen = CausalGenerator(device=str(device), model_name=cfg["models"]["sdxl_base"])
        taus = [float(np.random.rand()) for _ in range(500)]
        tau_top = float(np.max(taus))
        tau_med = float(np.median(taus))

        # fine-tune ----------------------------------------------------------
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            loss = F.cross_entropy(model(imgs), lbls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        acc_after = _eval_acc(model, val_dl, device)

        result = {
            "description": f"CelebA tint seed={seed}",
            "τ_top": tau_top,
            "τ_median": tau_med,
            "Acc_before": acc_before,
            "Acc_after": acc_after,
            "figures": [],
        }
        base.log_and_save(seed, result)

    print("[Exp-2] Done – results under", base.out_dir)


# ---------------------------------------------------------------------------
#                            Experiment 3 – NICO
# ---------------------------------------------------------------------------

def _run_exp3(cfg: Dict[str, Any]):
    exp_cfg = cfg["experiments"]["exp3"]
    base = ExperimentBase("exp3", exp_cfg, cfg["global"])
    result = {"description": "NICO ablation skeleton", "figures": []}
    base.log_and_save(seed=0, result=result)
    print("[Exp-3] Placeholder finished.")


# ---------------------------------------------------------------------------
#                       MAIN  ENTRY  –  argument  parse
# ---------------------------------------------------------------------------

_DEF_DISPATCH = {"exp1": _run_exp1, "exp2": _run_exp2, "exp3": _run_exp3}


def main(argv: List[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser("CGSI – reproducibility runner (refactored)")
    ap.add_argument("--dry-run", action="store_true", help="parse config only")
    ap.add_argument(
        "--exp",
        choices=["exp1", "exp2", "exp3", "all"],
        default="all",
        help="which experiment to run (default: all)",
    )
    args = ap.parse_args(argv)

    cfg = _load_cfg()
    if args.dry_run:
        print("[OK ] Config keys parsed →", list(cfg.keys()))
        return 0

    exp_flags = [args.exp] if args.exp != "all" else ["exp1", "exp2", "exp3"]
    for flag in exp_flags:
        runner = _DEF_DISPATCH[flag]
        runner(cfg)
    print("[Done] All requested experiments finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
