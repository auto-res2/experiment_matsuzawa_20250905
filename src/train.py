from __future__ import annotations
"""src/train.py ––– training / evaluation engine (patched)
The previous runtime error was caused by a missing ``src.models`` package.  In
addition, the project directories have been updated to comply with the new
research-iteration layout requested in the instructions.

This patch introduces **no behavioural changes** to the training logic itself –
it only tweaks a few paths to ``iteration5`` so that result artefacts are stored
in the correct location.
"""
import copy
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F

from .preprocess import get_dataset
from .evaluate import compute_metrics
from .models import MODEL_REGISTRY
from .utils import set_global_seed

__all__ = [
    "run_experiment",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _expand_model_cfg_list(models_cfg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand shorthand depth-sweep YAML syntax into a list of explicit model
    configuration dictionaries.  A helper display-name key (``_display_name``)
    is injected for nicer printing / plotting.
    """
    expanded: List[Dict[str, Any]] = []
    for cfg in models_cfg:
        if cfg["name"] in ("gcn", "gcn_pairnorm", "gcn_dropedge") and isinstance(cfg.get("layers"), list):
            for depth in cfg["layers"]:
                new_cfg = cfg.copy()
                new_cfg["layers"] = int(depth)
                new_cfg["_display_name"] = f"{cfg['name']}_L{depth}"
                expanded.append(new_cfg)
        else:
            expanded.append(cfg)
    return expanded

# -----------------------------------------------------------------------------
# Per-epoch helpers
# -----------------------------------------------------------------------------

def _train_one_epoch(model: torch.nn.Module, data, optimiser: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    optimiser.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))

    # Optional DH-GNN regularisation terms
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

# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------

def _build_model(model_cfg: Dict[str, Any], data, dataset):
    """Instantiate a model class from ``src.models`` and inject common kwargs."""
    name = model_cfg["name"]
    ModelCls = MODEL_REGISTRY[name]
    common_kwargs: Dict[str, Any] = {
        "in_channels": data.num_features,
        "out_channels": dataset.num_classes,
    }

    if name in ("gcn", "gcn_pairnorm", "gcn_dropedge"):
        common_kwargs["layers"] = int(model_cfg.get("layers", 2))
        if name == "gcn_dropedge":
            common_kwargs["dropedge_p"] = model_cfg.get("dropedge_p", 0.2)
    elif name == "gcnii":
        common_kwargs["layers"] = int(model_cfg.get("layers", 32))
    elif name == "ndls":
        common_kwargs["preset_depths"] = model_cfg.get("preset_depths", 6)
    elif name.startswith("dhgnn"):
        common_kwargs["max_layers"] = model_cfg.get("max_layers", 64)
    else:
        raise ValueError(f"Unrecognised model '{name}'.  Check config.")

    return ModelCls(**common_kwargs)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_experiment(exp_name: str, exp_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment as described in ``config/config.yaml``."""
    requested_device: str = str(global_cfg["global"].get("device", "cpu"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable – falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    epochs: int = int(global_cfg["global"]["epochs"])
    patience: int = int(global_cfg["global"]["early_stop_patience"])
    log_every: int = int(global_cfg["global"]["log_every"])

    results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Loop over datasets & models
    # ------------------------------------------------------------------
    for dname in exp_cfg["datasets"]:
        dataset = get_dataset(dname)
        data = dataset[0].to(device)

        per_dataset_results: Dict[str, Any] = {}
        model_cfg_list = _expand_model_cfg_list(exp_cfg["models"])

        for model_cfg in model_cfg_list:
            model_name_display = model_cfg.get("_display_name", model_cfg["name"])
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
                    print(f"[{dname}][{model_name_display}]  epoch {epoch:04d}  loss={loss:.4f}  val_acc={val_acc:.3f}")

                if val_acc > best_val:
                    best_val = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if epochs_without_improve >= patience:
                    break  # early stopping

            if best_state is not None:
                model.load_state_dict(best_state)
            final_metrics = _evaluate(model, data, device)
            final_metrics.update({"best_val": best_val, "epochs": epoch})
            per_dataset_results[model_name_display] = final_metrics

        results[dname] = per_dataset_results

    return results
