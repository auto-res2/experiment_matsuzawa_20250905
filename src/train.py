"""src/train.py
Model architecture construction and training utilities.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# -----------------------------------------------------------------------------
#                           MODEL  FACTORIES
# -----------------------------------------------------------------------------

def build_vit_s(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Return a ViT-S/16 backbone fine-tuned for *num_classes*."""
    # The timm 1.x API still supports the *pretrained* kwarg.
    return create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)

# -----------------------------------------------------------------------------
#                            GC-DRO  LOSS
# -----------------------------------------------------------------------------

class GCDROLoss(nn.Module):
    """Continuous Group-Conditional DRO loss implementation.

    Loss  =  CE * (1 + alpha · w_i)
    where *w_i* is a per-example weight.  A helper callable *weight_fn* can be
    attached dynamically so that the training loop stays generic.
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self._ce = nn.CrossEntropyLoss(reduction="none")
        # optional – will be populated by the main script when needed
        self.weight_fn = None  # type: ignore[attr-defined]

    def forward(  # type: ignore[override]
        self, logits: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor]
    ) -> torch.Tensor:
        per_example = self._ce(logits, labels)
        if weights is None:
            return per_example.mean()
        w = weights.float()
        # normalise such that E[w]=1 ⇒ stable LR comparison between ERM & DRO
        w = w / w.mean().clamp(min=1e-12)
        return (per_example * (1.0 + self.alpha * w)).mean()

    # tiny helper so that *self* can be used as a criterion from main.py
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
    loss_helper: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """Run **one** optimisation epoch and return (loss, accuracy)."""

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
                # compute per-example weights *only* if a weight function exists
                if hasattr(loss_helper, "weight_fn") and loss_helper.weight_fn is not None:  # type: ignore[attr-defined]
                    weights = loss_helper.weight_fn(group_idx.cuda())  # type: ignore[arg-type,operator]
                else:
                    weights = None
                loss = loss_helper.criterion(logits, labels, weights)  # type: ignore[attr-defined]

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return loss_sum / total, correct / total
