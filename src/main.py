"""src/main.py
Entry-point orchestrating the full experimental sweep.  Execute with
    python -m src.main
from the project root.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import yaml

from .preprocess import DatasetFactory, make_loader, set_seed
from .train import (
    CounterfactualGenerator,
    ERMTrainer,
    GroupDROTrainer,
    PCDTrainer,
    build_model,
    get_device,
)
from .evaluate import IMG_DIR, RES_DIR, plot_learning_curves, save_results_json

# -----------------------------------------------------------------------------
# CONFIG ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG: Dict = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())


# -----------------------------------------------------------------------------
# SINGLE RUN -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def run_single_experiment(dataset_key: str, model_cfg: Dict, algo: str, seed: int):
    print(f"Running {algo} on {dataset_key} | model={model_cfg['name']} | seed={seed}")
    set_seed(seed)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if dataset_key == "cifar10":
        train_ds, val_ds, test_ds, meta = DatasetFactory.cifar10()
    elif dataset_key == "waterbirds":
        train_ds, val_ds, test_ds, meta = DatasetFactory.waterbirds()
    elif dataset_key == "celeba":
        train_ds, val_ds, test_ds, meta = DatasetFactory.celeba()
    else:
        raise ValueError(f"Unsupported dataset {dataset_key}.")

    has_group = meta.get("group_labels_present", False)
    bsz = CONFIG["global"]["batch_size"]
    train_loader = make_loader(train_ds, bsz, shuffle=True)
    val_loader = make_loader(val_ds, bsz, shuffle=False)
    test_loader = make_loader(test_ds, bsz, shuffle=False)

    # ------------------------------------------------------------------
    # Model & Trainer
    # ------------------------------------------------------------------
    model = build_model(model_cfg, meta["num_classes"])
    optim_cfg = {
        "lr": CONFIG["global"]["optim"]["lr_resnet"] if model_cfg["type"] == "resnet" else CONFIG["global"]["optim"]["lr_vit"],
        "weight_decay": CONFIG["global"]["optim"]["weight_decay"],
        "epochs": CONFIG["global"]["epochs"][dataset_key.split("_")[0]],
        "lambda_inv": CONFIG["global"]["loss_weights"]["lambda_inv"],
        "lambda_feat": CONFIG["global"]["loss_weights"]["lambda_feat"],
    }

    if algo == "GroupDRO" and not has_group:
        print("[WARN] Group labels not present – falling back to ERM.")
        algo = "ERM"

    if algo == "ERM":
        trainer = ERMTrainer(model, optim_cfg, meta)
    elif algo == "GroupDRO":
        trainer = GroupDROTrainer(model, optim_cfg, meta)
    elif algo == "PCD":
        generator = CounterfactualGenerator(CONFIG["global"]["counterfactual"])
        trainer = PCDTrainer(model, optim_cfg, meta, generator)
    else:
        raise ValueError(f"Unknown algorithm {algo}.")

    # ------------------------------------------------------------------
    # Fit & Evaluate
    # ------------------------------------------------------------------
    best_val, history = trainer.fit(train_loader, val_loader)
    test_acc = trainer.evaluate(test_loader)

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    result_dict = {
        "dataset": dataset_key,
        "model": model_cfg["name"],
        "algorithm": algo,
        "seed": seed,
        "val_accuracy": best_val,
        "test_accuracy": test_acc,
    }

    json_name = f"{dataset_key}_{model_cfg['name']}_{algo}_seed{seed}.json"
    save_results_json(result_dict, RES_DIR / json_name)

    title = f"Training Loss – {algo} on {dataset_key} ({model_cfg['name']})"
    fig_name = IMG_DIR / f"training_loss_{dataset_key}_{model_cfg['name']}_{algo}.pdf"
    plot_learning_curves(history["train_loss"], title, fig_name)

    # stdout for CI / manual inspection
    print("\n==================== SUMMARY ====================")
    print(yaml.safe_dump(result_dict, sort_keys=False))
    print("Figure saved:", fig_name.relative_to(ROOT))
    sys.stdout.flush()


# -----------------------------------------------------------------------------
# MAIN LOOP --------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():  # noqa: D401
    datasets_to_run = ["cifar10", "waterbirds", "celeba"]
    for dataset_key in datasets_to_run:
        for model_cfg in CONFIG["models"]:
            for algo in CONFIG["algorithms"]:
                for seed in CONFIG["global"]["seeds"]:
                    run_single_experiment(dataset_key, model_cfg, algo, seed)


if __name__ == "__main__":
    main()