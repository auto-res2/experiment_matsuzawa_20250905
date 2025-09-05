from __future__ import annotations

"""main.py – orchestrates the whole experimental workflow via relative imports."""

from pathlib import Path
import yaml
import json

from .preprocess import set_seed, SEEDS, load_dataset
from .train import AttributeMiner, Trainer
from .evaluate import save_lineplot

# --------------------------------------------------------------------------- #
# === configuration ========================================================= #
# --------------------------------------------------------------------------- #
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Cannot locate experiment configuration at {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# === helpers =============================================================== #
# --------------------------------------------------------------------------- #

def run_experiment(exp_cfg):
    print("\n" + "=" * 79)
    print(f"Experiment: {exp_cfg['id']}")
    print(exp_cfg["description"])
    print("=" * 79 + "\n")

    train_ds, val_ds = load_dataset(exp_cfg)

    # ---- spurious-attribute mining (unsupervised) ------------------------ #
    miner = AttributeMiner(k=exp_cfg["k_clusters"])
    cluster_ids, _ = miner.mine(train_ds)
    print(f"[Miner] derived {len(set(cluster_ids))} clusters for {len(cluster_ids)} samples.")

    # ---- training -------------------------------------------------------- #
    trainer = Trainer(exp_cfg, train_ds, val_ds)
    metrics = trainer.run()

    # ---- plotting -------------------------------------------------------- #
    fig_name = save_lineplot(metrics["val_acc"], "Validation accuracy", "acc", f"accuracy_{exp_cfg['id']}")

    print("\n----- SUMMARY -----")
    print(json.dumps(metrics, indent=2))
    print("Generated figure:", fig_name)
    print("-------------------\n")


# --------------------------------------------------------------------------- #
# === entry point =========================================================== #
# --------------------------------------------------------------------------- #

def main():  # noqa: D401 – script entry point
    for exp_cfg in CONFIG["experiments"]:
        for seed in SEEDS:
            print(f"\n>>> Running {exp_cfg['id']}  seed={seed}")
            set_seed(seed)
            run_experiment(exp_cfg)


if __name__ == "__main__":
    main()