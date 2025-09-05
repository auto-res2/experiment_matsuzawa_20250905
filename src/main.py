from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .preprocess import (
    CFG,
    apply_meta_attack,
    get_config,
    load_dataset,
    mask_features,
    set_seed,
)
from .train import (
    AFNGNN,
    ChebFilterBank,
    ChebGCN,
    GCNIIWrapper,
    train_one,
)
from .evaluate import plot_line

# fixed image directory -------------------------------------------------------
IMG_DIR = Path(".research/iteration2/images")  # Updated path
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
#  Experiment 1 – depth scalability
# ----------------------------------------------------------------------------

def experiment_depth_scalability(cfg):
    print("\nEXPERIMENT 1 — DEPTH-SCALABILITY BENCHMARK")
    results: Dict[str, Dict[str, List[float]]] = {}
    depth_grid = cfg["experiments"]["depth_scalability"]["depths"]

    for dataset_name in cfg["experiments"]["depth_scalability"]["datasets"]:
        data = load_dataset(dataset_name)
        results[dataset_name] = {m: [] for m in ["AFN", "ChebGCN", "GCNII"]}
        print(f"\nDataset: {dataset_name}")
        bank = ChebFilterBank(data, cfg["model"]["bands"])
        for depth in depth_grid:
            hidden = (
                cfg["model"]["hidden_dim_large"] if depth <= 32 else cfg["model"]["hidden_dim_small"]
            )
            # AFN ----------------------------------------------------------------
            afn_accs = []
            for seed in cfg["seed_list"]:
                set_seed(seed)
                model = AFNGNN(
                    data.num_features,
                    hidden,
                    int(data.y.max()) + 1,
                    depth,
                    cfg["model"]["bands"],
                    cfg["model"]["dropout"],
                )
                test_acc, _, _ = train_one(model, data, bank, cfg)
                afn_accs.append(test_acc)
            mean_afn = float(np.mean(afn_accs))
            results[dataset_name]["AFN"].append(mean_afn)
            print(f"  Depth {depth:3d} — AFN  acc: {mean_afn*100:.2f}%")

            # ChebGCN ------------------------------------------------------------
            cheb_accs = []
            for seed in cfg["seed_list"]:
                set_seed(seed)
                model = ChebGCN(
                    data.num_features,
                    hidden,
                    int(data.y.max()) + 1,
                    depth,
                    cfg["model"]["bands"],
                    cfg["model"]["dropout"],
                )
                test_acc, _, _ = train_one(model, data, bank, cfg)
                cheb_accs.append(test_acc)
            results[dataset_name]["ChebGCN"].append(float(np.mean(cheb_accs)))

            # GCNII --------------------------------------------------------------
            gcn_accs = []
            repo_url = cfg["resources"]["gcnII_repo"]
            for seed in cfg["seed_list"]:
                set_seed(seed)
                model = GCNIIWrapper(
                    data.num_features,
                    hidden,
                    int(data.y.max()) + 1,
                    depth,
                    cfg["model"]["dropout"],
                    repo_url=repo_url,
                )
                test_acc, _, _ = train_one(model, data, bank, cfg)
                gcn_accs.append(test_acc)
            results[dataset_name]["GCNII"].append(float(np.mean(gcn_accs)))

        # per-dataset plot -----------------------------------------------------
        fig = f"accuracy_depth_{dataset_name.lower()}.pdf"
        plot_line(
            depth_grid,
            {k: [v * 100 for v in vs] for k, vs in results[dataset_name].items()},
            title=f"Accuracy vs Depth — {dataset_name}",
            xlabel="Depth (layers)",
            ylabel="Accuracy (%)",
            filename=fig,
        )

    # print summary -----------------------------------------------------------
    print("\nAggregated Accuracy (mean over seeds)")
    for ds, res in results.items():
        print(f"  {ds}:")
        for model_name, accs in res.items():
            formatted = ", ".join(f"{a*100:.2f}" for a in accs)
            print(f"    {model_name:<7}: {formatted}")


# ----------------------------------------------------------------------------
#  Experiment 2 – node-wise frequency usage (Chameleon)
# ----------------------------------------------------------------------------

def experiment_frequency_usage(cfg):
    print("\nEXPERIMENT 2 — NODE-WISE FREQUENCY-USAGE ANALYSIS")
    dataset_name = cfg["experiments"]["frequency_usage"]["dataset"]
    data = load_dataset(dataset_name)
    depth = 32
    hidden = cfg["model"]["hidden_dim_large"]
    bank = ChebFilterBank(data, cfg["model"]["bands"])

    seed = cfg["seed_list"][0]
    set_seed(seed)
    model = AFNGNN(
        data.num_features,
        hidden,
        int(data.y.max()) + 1,
        depth,
        cfg["model"]["bands"],
        cfg["model"]["dropout"],
    )
    train_one(model, data, bank, cfg)
    model.eval()

    with torch.no_grad():
        _, _, all_weights = model(data, bank)

    layers_to_probe = cfg["experiments"]["frequency_usage"]["layers_to_probe"]
    deg = data.deg.numpy()
    curv = data.node_curv.numpy()
    for l in layers_to_probe:
        w = all_weights[l - 1].numpy()
        high_mass = w[:, -1]
        r_deg = np.corrcoef(high_mass, 1.0 / (deg + 1e-5))[0, 1]
        r_curv = np.corrcoef(high_mass, np.abs(curv))[0, 1]
        print(f"Layer {l:2d}: corr(high_mass, 1/deg) = {r_deg:.3f}, corr(high_mass, |curv|) = {r_curv:.3f}")

    # scatter plot -----------------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 4))
    plt.scatter(np.abs(curv), high_mass, alpha=0.4, s=8)
    plt.xlabel("|Ricci curvature|")
    plt.ylabel("High-frequency weight (k=K)")
    plt.title("Layer-32 gate weight vs curvature")
    fig_name = IMG_DIR / "highfreq_vs_curvature.pdf"
    plt.tight_layout()
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fig_name}")


# ----------------------------------------------------------------------------
#  Experiment 3 – robustness
# ----------------------------------------------------------------------------

def experiment_robustness(cfg):
    print("\nEXPERIMENT 3 — ROBUSTNESS AGAINST ATTACKS")
    edge_budgets = cfg["experiments"]["robustness"]["edge_budgets"]
    feat_budgets = cfg["experiments"]["robustness"]["feat_budgets"]

    for dataset_name in cfg["experiments"]["robustness"]["datasets"]:
        data_clean = load_dataset(dataset_name)
        bank_clean = ChebFilterBank(data_clean, cfg["model"]["bands"])
        seed = cfg["seed_list"][0]
        set_seed(seed)
        model_clean = AFNGNN(
            data_clean.num_features,
            cfg["model"]["hidden_dim_large"],
            int(data_clean.y.max()) + 1,
            depth=32,
            K=cfg["model"]["bands"],
            dropout=cfg["model"]["dropout"],
        )
        clean_acc, _, _ = train_one(model_clean, data_clean, bank_clean, cfg)
        print(f"\nDataset {dataset_name} — clean accuracy: {clean_acc*100:.2f}%")

        # edge perturbations --------------------------------------------------
        drops = []
        for ptb in edge_budgets:
            data_ptb = apply_meta_attack(data_clean, ptb, seed)
            bank_ptb = ChebFilterBank(data_ptb, cfg["model"]["bands"])
            model_ptb = AFNGNN(
                data_ptb.num_features,
                cfg["model"]["hidden_dim_large"],
                int(data_ptb.y.max()) + 1,
                depth=32,
                K=cfg["model"]["bands"],
                dropout=cfg["model"]["dropout"],
            )
            acc_ptb, _, _ = train_one(model_ptb, data_ptb, bank_ptb, cfg)
            delta = clean_acc - acc_ptb
            drops.append(delta)
            print(f"  MetaAttack {ptb*100:.0f}% — ΔAcc = {delta*100:.2f} pp")
        plot_line(
            [e * 100 for e in edge_budgets],
            {"AFN": [d * 100 for d in drops]},
            title=f"Accuracy drop vs Edge perturbation — {dataset_name}",
            xlabel="Perturbation (%)",
            ylabel="ΔAccuracy (pp)",
            filename=f"robustness_edge_{dataset_name.lower()}.pdf",
        )

        # feature masking -----------------------------------------------------
        drops_feat = []
        for mr in feat_budgets:
            data_mask = mask_features(data_clean, mr, seed)
            bank_mask = ChebFilterBank(data_mask, cfg["model"]["bands"])
            model_mask = AFNGNN(
                data_mask.num_features,
                cfg["model"]["hidden_dim_large"],
                int(data_mask.y.max()) + 1,
                depth=32,
                K=cfg["model"]["bands"],
                dropout=cfg["model"]["dropout"],
            )
            acc_mask, _, _ = train_one(model_mask, data_mask, bank_mask, cfg)
            delta_feat = clean_acc - acc_mask
            drops_feat.append(delta_feat)
            print(f"  Feature mask {mr*100:.0f}% — ΔAcc = {delta_feat*100:.2f} pp")
        plot_line(
            [f * 100 for f in feat_budgets],
            {"AFN": [d * 100 for d in drops_feat]},
            title=f"Accuracy drop vs Feature masking — {dataset_name}",
            xlabel="Mask ratio (%)",
            ylabel="ΔAccuracy (pp)",
            filename=f"robustness_feat_{dataset_name.lower()}.pdf",
        )


# ----------------------------------------------------------------------------
#  Main entry point
# ----------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")
    cfg = get_config()

    start = time.time()
    experiment_depth_scalability(cfg)
    experiment_frequency_usage(cfg)
    experiment_robustness(cfg)
    elapsed = (time.time() - start) / 3600
    print(f"\nAll experiments finished in {elapsed:.2f} hours.")


if __name__ == "__main__":
    main()
