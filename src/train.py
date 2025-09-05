from __future__ import annotations
"""src/train.py ––– training / evaluation engine
This module contains the high-level experiment loop and the per-epoch
training / evaluation helpers.  It orchestrates model construction,
optimiser / scheduler creation and implements early stopping.  Heavy
number-crunching happens here; no file-system side-effects occur except
for model-checkpoint storage via ``torch.save`` when desired.
"""
import copy
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F

from .preprocess import get_dataset
from .evaluate import compute_metrics
from .models import MODEL_REGISTRY
from .utils import set_global_seed

__all__ = [
    "run_experiment",
]

def _train_one_epoch(model: torch.nn.Module,
                     data,
                     optimiser: torch.optim.Optimizer,
                     device: torch.device) -> float:
    """Forward / backward pass for **one** epoch.  Additional loss terms
    (e.g. DH-GNN regularisers) are attached automatically if present."""
    model.train()
    optimiser.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))

    # Optional DH-GNN regularisers
    if hasattr(model, "regularisation") and isinstance(model.regularisation, dict):
        regs = model.regularisation
        loss = loss + regs.get("mdr", 0.0) + regs.get("budget", 0.0)

    loss.backward()
    optimiser.step()
    return float(loss.item())


@torch.no_grad()
def _evaluate(model: torch.nn.Module, data, device: torch.device):
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    return compute_metrics(logits, data)


def _build_model(model_cfg: Dict[str, Any], data, dataset):
    """Instantiate a model from the registry, inferring input / output sizes
    and passing through model-specific keyword arguments."""
    name = model_cfg["name"]
    ModelCls = MODEL_REGISTRY[name]
    common_kwargs: Dict[str, Any] = {
        "in_channels": data.num_features,
        "out_channels": dataset.num_classes,
    }

    # ------------------------------------------------------------------
    # Model-specific argument dispatch
    # ------------------------------------------------------------------
    if name in ("gcn", "gcn_pairnorm", "gcn_dropedge"):
        common_kwargs["layers"] = model_cfg.get("layers", 2)
        if name == "gcn_dropedge":
            common_kwargs["dropedge_p"] = model_cfg.get("dropedge_p", 0.2)
    elif name == "gcnii":
        common_kwargs["layers"] = model_cfg.get("layers", 32)
    elif name == "ndls":
        common_kwargs["preset_depths"] = model_cfg.get("preset_depths", 6)
    elif name.startswith("dhgnn"):
        common_kwargs["max_layers"] = model_cfg.get("max_layers", 64)
    else:
        raise ValueError(f"Unrecognised model '{name}'.  Check config.")

    return ModelCls(**common_kwargs)


def run_experiment(exp_name: str,
                   exp_cfg: Dict[str, Any],
                   global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run **one** experiment as specified in ``config/config.yaml``.

    Parameters
    ----------
    exp_name:  Human-readable experiment identifier (used for JSON / figure names).
    exp_cfg:   Experiment-specific sub-dictionary from the YAML config.
    global_cfg:Full, already-loaded YAML configuration dict.
    """
    device = torch.device(global_cfg["global"]["device"])
    epochs: int = int(global_cfg["global"]["epochs"])
    patience: int = int(global_cfg["global"]["early_stop_patience"])
    log_every: int = int(global_cfg["global"]["log_every"])

    results: Dict[str, Any] = {}

    for dname in exp_cfg["datasets"]:
        dataset = get_dataset(dname)
        data = dataset[0].to(device)

        per_dataset_results: Dict[str, Any] = {}
        for model_cfg in exp_cfg["models"]:
            model_name = model_cfg["name"]
            model = _build_model(model_cfg, data, dataset).to(device)
            optimiser = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

            best_val: float = 0.0
            best_state = None
            epochs_without_improve = 0
            history = {"train_loss": [], "val_acc": []}

            for epoch in range(1, epochs + 1):
                loss = _train_one_epoch(model, data, optimiser, device)
                metrics = _evaluate(model, data, device)
                val_acc = float(metrics["val_acc"])

                history["train_loss"].append(loss)
                history["val_acc"].append(val_acc)

                if epoch % log_every == 0:
                    print(f"[{dname}][{model_name}]  epoch {epoch:04d}  loss={loss:.4f}  val_acc={val_acc:.3f}")

                if val_acc > best_val:
                    best_val = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if epochs_without_improve >= patience:
                    break  # early stopping

            # Restore best checkpoint & perform final evaluation on the test set
            if best_state is not None:
                model.load_state_dict(best_state)
            final_metrics = _evaluate(model, data, device)
            final_metrics.update({"best_val": best_val, "epochs": epoch})
            per_dataset_results[model_name] = final_metrics

        results[dname] = per_dataset_results

    return results
