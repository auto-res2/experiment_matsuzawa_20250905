"""src/train.py
Model architecture construction and training utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# -----------------------------------------------------------------------------
#                           MODEL  FACTORIES
# -----------------------------------------------------------------------------

def build_vit_s(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Factory that returns a ViT-S/16 backbone fine-tuned for *num_classes*.
    The function wraps `timm.create_model` so that we have a *single* place where
    model hyper-parameters are set.  This makes it easier to swap backbones from
    the main script without touching the rest of the code base.
    """
    model = create_model(
        "vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes
    )
    return model

# -----------------------------------------------------------------------------
#                            GC-DRO  LOSS
# -----------------------------------------------------------------------------

class GCDROLoss(nn.Module):
    """Continuous Group-Conditional DRO loss.

    Loss  =  CE * (1 + alpha * w_i)
    where *w_i* is a per-instance weight indicating how much an example belongs
    to a (potentially soft) minority group.  The *weights* tensor must already
    live on the same device as the *logits* that enter the forward method.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self._ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor]
    ) -> torch.Tensor:  # type: ignore[override]
        per_example = self._ce(logits, labels)
        if weights is None:
            return per_example.mean()
        w = weights.float()
        # normalise such that E[w]=1.  This keeps the learning-rate tuning stable
        # when we move from ERM (all w=0) to DRO (some w>0).
        w = w / w.mean().clamp(min=1e-12)
        return (per_example * (1.0 + self.alpha * w)).mean()

    # a tiny helper so that *self* can be re-used as a criterion from main.py
    def criterion(self, logits, labels, weights):  # pylint: disable=method-hidden
        return self.forward(logits, labels, weights)

# -----------------------------------------------------------------------------
#                                TRAINING
# -----------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    loss_helper: Optional[GCDROLoss] = None,
) -> Tuple[float, float]:
    """Runs **one** optimisation epoch.

    Returns
    -------
    Tuple[float, float]
        (mean training loss, training accuracy)
    """
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels, group_idx in loader:
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            if loss_helper is None:
                loss = F.cross_entropy(logits, labels)
            else:
                # *group_idx* is still on CPU – move only when required to save PCIe time
                weights = loss_helper(group_idx.cuda()) if callable(loss_helper) else None
                loss = loss_helper.criterion(logits, labels, weights)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return loss_sum / total, correct / total
