"""src.main – orchestrates the whole experimental workflow with the refactored
code-base.  Execute via `python -m src.main` from the repository root."""
from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.backends.cudnn as cudnn
import yaml

from .train import DFBDModel
from .preprocess import (
    assert_imagenet_present,
    imagenet_task_dataloaders,
)
from .evaluate import (
    evaluate,
    set_seed,
    plot_line,
    save_json,
)

# -----------------------------------------------------------------------------
# Load YAML configuration ------------------------------------------------------
# -----------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

cudnn.benchmark = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

# -----------------------------------------------------------------------------
# Experiment-1 (ImageNet-128) --------------------------------------------------
# -----------------------------------------------------------------------------

def run_experiment1(seed: int, budget_kb: int) -> Dict[str, float]:
    desc = CONFIG["experiments"]["exp1"]["description"]
    print(f"\n===== Experiment-1 – {desc} | Budget: {budget_kb} kB | Seed {seed} =====")

    set_seed(seed)
    root = Path(CONFIG["datasets"]["imagenet128"]["root"])
    assert_imagenet_present(root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFBDModel(
        K=CONFIG["models"]["dfbd"]["K"],
        rho=CONFIG["models"]["dfbd"]["rho"],
        bits_per_basis=CONFIG["models"]["dfbd"]["bits_per_basis"],
        num_classes=1000,
    ).to(device)

    opt_cfg = CONFIG["global"]["optimiser"]
    optimiser = torch.optim.AdamW(model.parameters(), **opt_cfg)

    scaler = (
        torch.cuda.amp.GradScaler(enabled=CONFIG["global"]["mixed_precision"])
        if torch.cuda.is_available()
        else None
    )

    n_tasks = CONFIG["datasets"]["imagenet128"]["n_tasks"]
    batch_size = CONFIG["global"]["batch_size"]

    acc_per_task: List[float] = []
    mem_curve: List[int] = []

    for task_id in range(n_tasks):
        loader, _ = imagenet_task_dataloaders(
            root,
            CONFIG["datasets"]["imagenet128"]["img_size"],
            task_id,
            batch_size,
        )
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=CONFIG["global"]["mixed_precision"]):
                loss = model.forward_and_loss(x, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt_cfg["grad_clip"]
                )
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt_cfg["grad_clip"]
                )
                optimiser.step()

            if model.should_self_compress():
                model.self_compress()

            mem_curve.append(model.byte_size())
            if model.byte_size() > budget_kb * 1024:
                raise RuntimeError(
                    f"DFBD memory {model.byte_size()/1024:.1f} kB exceeds budget {budget_kb} kB"
                )

        # --- validation after task ------------------------------------
        val_loader, _ = imagenet_task_dataloaders(
            root,
            CONFIG["datasets"]["imagenet128"]["img_size"],
            task_id,
            batch_size=256,
        )
        acc = evaluate(model, val_loader, device)
        acc_per_task.append(acc)
        print(
            f"Task {task_id:3d} finished – Val Acc: {acc*100:.2f}% | "
            f"Replay mem: {model.byte_size()/1024:.1f} kB"
        )

    avg_acc = sum(acc_per_task) / len(acc_per_task)
    forgetting = max(acc_per_task) - acc_per_task[-1]

    fig_path = plot_line(
        list(range(len(mem_curve))),
        {"DFBD": [m / 1024 for m in mem_curve]},
        title="Replay memory vs iteration",
        xlabel="Iteration",
        ylabel="Memory (kB)",
        filename="memory_curve_dfbd.pdf",
    )
    print("Figure generated:", fig_path.name)

    results = {
        "seed": seed,
        "budget_kb": budget_kb,
        "avg_accuracy": avg_acc,
        "forgetting": forgetting,
    }

    out_dir = Path(".research") / "iteration1"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"experiment1_seed{seed}_budget{budget_kb}.json"
    save_json(results, res_path)

    print("Experiment-1 results (JSON):")
    print(json.dumps(results, indent=2))
    return results

# -----------------------------------------------------------------------------
# Main dispatcher --------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    results_all: List[Dict[str, float]] = []
    try:
        for seed in CONFIG["global"]["seeds"]:
            for budget in CONFIG["experiments"]["exp1"]["memory_budgets_kb"]:
                res = run_experiment1(seed, budget)
                results_all.append(res)
    except FileNotFoundError as e:
        print("\nERROR:", e)
        sys.exit(1)

    if results_all:
        summary_path = Path(".research") / "iteration1" / "exp1_summary.json"
        save_json({"all": results_all}, summary_path)
        print("\n===== All Experiment-1 runs completed successfully =====")
        print(json.dumps({"all": results_all}, indent=2))


if __name__ == "__main__":
    main()
