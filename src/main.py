"""src/main.py
Entry point orchestrating the complete EXP-1 (Waterbirds) pipeline.
Run via:
    python -m src.main
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import torch
import yaml
from fvcore.nn import FlopCountAnalysis

from .evaluate import evaluate, evaluate_pgd, save_lineplot
from .preprocess import build_dataloaders, set_seed, _download_waterbirds
from .train import GCDROLoss, build_vit_s, train_epoch

# -----------------------------------------------------------------------------
#                           CONFIG  LOADING
# -----------------------------------------------------------------------------

CONFIG_PATH = Path("config/config.yaml")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        "Configuration file not found – you must create config/config.yaml before running."
    )
CFG: Dict = yaml.safe_load(CONFIG_PATH.read_text())

# Override paths to comply with the task requirements -------------------------
RESULTS_DIR = Path(".research/iteration58")
FIG_DIR = RESULTS_DIR / "images"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#                              EXPERIMENT
# -----------------------------------------------------------------------------

def _run_single_seed(seed: int) -> None:
    print(f"\n===== Running seed {seed} =====")
    set_seed(seed)

    # ----------------------- DATA -------------------------------------------
    _download_waterbirds(CFG["dataset_repo"], CFG["data_root"])
    ld_train, ld_val, ld_test = build_dataloaders(CFG)

    # -------------------- 3 baselines ---------------------------------------
    results_json: Dict[str, Dict] = {}

    for method in ["ERM", "GC-DRO-Oracle", "CGSI"]:
        print(f"\n[Model] training {method}")
        model = build_vit_s().cuda()
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
        )
        scaler = torch.cuda.amp.GradScaler()

        # GC-DRO helper -------------------------------------------------------
        if method == "GC-DRO-Oracle":
            dro_loss = GCDROLoss(alpha=0.1)

            # minority groups are 1 & 2 (Waterbirds convention)
            def _weights(gidx: torch.Tensor) -> torch.Tensor:  # noqa: N801 – local helper
                return ((gidx == 1) | (gidx == 2)).float()

            dro_loss.weight_fn = _weights  # type: ignore[attr-defined]
            loss_helper = dro_loss
        else:
            loss_helper = None

        epoch_acc, epoch_wg, epoch_gap, times = [], [], [], []

        for ep in range(1, CFG["epochs"] + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, ld_train, optimiser, scaler, loss_helper)
            val_acc, val_wg, val_gap, _ = evaluate(model, ld_val)
            times.append(time.time() - t0)

            epoch_acc.append(val_acc)
            epoch_wg.append(val_wg)
            epoch_gap.append(val_gap)

            flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).cuda()).total() / 1e9
            row = {
                "epoch": ep,
                "sec": round(times[-1], 2),
                "train_loss": train_loss,
                "val_AvgAcc": val_acc,
                "val_WGAcc": val_wg,
                "val_CorrGap": val_gap,
                "FLOPs(G)": round(flops, 2),
            }
            print("[Epoch]", json.dumps(row))

        # ----------------- final test set ------------------------------------
        test_acc, test_wg, test_gap, _ = evaluate(model, ld_test)
        pgd_acc = evaluate_pgd(model, ld_test, eps=CFG["pgd_eps"], steps=CFG["pgd_steps"])

        results_json[method] = {
            "method": method,
            "seed": seed,
            "AvgAcc": round(test_acc, 4),
            "WGAcc": round(test_wg, 4),
            "CorrGap": round(test_gap, 4),
            "PGDAcc": round(pgd_acc, 4),
            "train_sec_total": round(sum(times), 2),
        }

        xs = list(range(1, CFG["epochs"] + 1))
        save_lineplot(xs, {"AvgAcc": epoch_acc, "WGAcc": epoch_wg}, "Accuracy", f"{method}_acc_curve", FIG_DIR)
        save_lineplot(xs, {"CorrGap": epoch_gap}, "CorrGap", f"{method}_cor_gap", FIG_DIR)

    # ---------------------- persist JSON ------------------------------------
    out_json = RESULTS_DIR / f"results_seed{seed}.json"
    out_json.write_text(json.dumps(results_json, indent=2))
    print(f"\n[JSON] {out_json} =\n{json.dumps(results_json, indent=2)}")


def main() -> None:
    print("================ EXP-1  WATERBIRDS  ==================")
    print("Full hyper-parameters:")
    print(json.dumps(CFG, indent=2))

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required for this experiment.")

    for s in CFG["seeds"]:
        _run_single_seed(s)

    print("\nAll seeds finished – raw JSON + figures saved in", RESULTS_DIR)


if __name__ == "__main__":  # pragma: no cover
    main()
