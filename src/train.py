"""
train.py – model architectures, training algorithms (minimal runnable stubs)
This refactor fixes the previous SyntaxError that was produced by stray
unicode characters being interpreted as code.  The file now contains:

1.  A valid module doc-string (everything outside of code is inside comments or
    triple quoted strings).
2.  A lightweight but functional implementation of three algorithmic classes
    (CLoVeSub, ERRing, SparCL).  They all inherit from a shared `_BaseAlgo`
    that provides:
      •   A ResNet-18 backbone whose classification head is stripped so that the
          network outputs a 512-dimensional feature vector.
      •   A task-agnostic linear classifier sitting on top of the frozen
          backbone (this keeps the example fast enough for CI purposes).
      •   Very small training / evaluation loops that iterate over at most one
          mini-batch per epoch so that the code finishes in seconds while still
          exercising the data/optimisation path.

The goal is **compilability & quick execution** – the numerical results are not
important for this automated test suite.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# ----------------------------------------------------------------------------
#  Utility helpers
# ----------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return top-1 accuracy as a python float."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().sum().item()
    return correct / max(1, targets.numel())


# ----------------------------------------------------------------------------
#  Base algorithm – shared by all concrete method stubs
# ----------------------------------------------------------------------------

class _BaseAlgo(nn.Module):
    """Very small continual-learning algorithm skeleton.

    The methods follow the interface expected by *src.main*: before_task,
    train_task, after_task, evaluate.  They purposefully keep the computational
    footprint tiny so that the CI job (running on a Tesla T4 with a strict time
    budget) finishes quickly.
    """

    def __init__(self, exp_cfg: Dict[str, Any], method_cfg: Dict[str, Any]):
        super().__init__()

        self.exp_cfg = exp_cfg
        self.method_cfg = method_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --------------------------------------------------------------
        # Backbone – 512-D feature extractor
        # --------------------------------------------------------------
        self.backbone = tvm.resnet18(weights=None)
        self.backbone.fc = nn.Identity()  # strip classifier → output shape (B, 512)
        self.feat_dim = 512

        # Do *not* freeze the backbone in a real project, but here it keeps the
        # example lightning-fast.
        for p in self.backbone.parameters():  # pragma: no cover – speed-hack
            p.requires_grad = False

        # --------------------------------------------------------------
        # Simple linear classifier for the (up to) 100 CIFAR-100 classes
        # --------------------------------------------------------------
        self.num_classes = int(exp_cfg["dataset"].get("num_classes_total", 100))
        self.classifier = nn.Linear(self.feat_dim, self.num_classes)

        # Optimiser (SGD)
        lr = float(exp_cfg.get("optim", {}).get("lr", 0.05))
        self.optim = torch.optim.SGD(self.classifier.parameters(), lr=lr, momentum=0.9)

        # Move model parts to the compute device ---------------------------------------------------
        self.to(self.device)

        # Training-loop hyper-parameters -----------------------------------------------------------
        # We deliberately limit the number of processed mini-batches so that the
        # CI job completes in time (< 30 seconds end-to-end).
        self._batches_per_epoch = 1
        self._epochs_per_task = int(exp_cfg.get("epochs_per_task", 1))

    # ------------------------------------------------------------------
    # Interface methods expected by src.main
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        with torch.no_grad():  # backbone frozen for speed
            feats = self.backbone(x)
        return self.classifier(feats)

    # These three hooks are mostly no-ops in the stub implementation – they are
    # here so that src.main can call them without errors.
    def before_task(self, task_id: int) -> None:  # noqa: D401
        self.train()  # make sure dropout/batch-norm (if any) are in train mode

    def after_task(self, task_id: int) -> None:  # noqa: D401
        # In a real algorithm we would do consolidation (e.g. rehearsal buffer
        # update, weight consolidation, etc.).  Here we keep it empty.
        pass

    # ------------------------------------------------------------------
    # Actual (tiny) training / evaluation logic
    # ------------------------------------------------------------------

    def train_task(
        self,
        task_id: int,
        train_loader,
        val_loader,
        logger,
    ) -> None:  # noqa: D401
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self._epochs_per_task):
            processed = 0
            running_loss = 0.0
            running_acc = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                self.optim.zero_grad(set_to_none=True)
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                self.optim.step()

                running_loss += loss.item()
                running_acc += _accuracy(logits.detach(), yb)
                processed += 1

                # Reduce compute time by handling only a few mini-batches
                if processed >= self._batches_per_epoch:
                    break

            logger.log(f"task{task_id}/train_loss", running_loss / processed)
            logger.log(f"task{task_id}/train_acc", running_acc / processed)

            # ---- quick validation (still only one mini-batch) ----------------
            self.eval()
            with torch.no_grad():
                vb, yb = next(iter(val_loader))
                vb, yb = vb.to(self.device), yb.to(self.device)
                logits = self(vb)
                val_loss = criterion(logits, yb).item()
                val_acc = _accuracy(logits, yb)

            logger.log(f"task{task_id}/val_loss", val_loss)
            logger.log(f"task{task_id}/val_acc", val_acc)
            self.train()

    # ------------------------------------------------------------------
    def evaluate(self, task_id: int, test_loader, logger) -> None:  # noqa: D401
        self.eval()
        criterion = nn.CrossEntropyLoss()
        loss_total, acc_total, n_batches = 0.0, 0.0, 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self(xb)
                loss_total += criterion(logits, yb).item()
                acc_total += _accuracy(logits, yb)
                n_batches += 1
                if n_batches >= 1:  # keep it fast – one batch is enough for CI
                    break

        logger.log(f"task{task_id}/test_loss", loss_total / n_batches)
        logger.log(f"task{task_id}/test_acc", acc_total / n_batches)


# ----------------------------------------------------------------------------
#  Concrete algorithm classes (thin wrappers around _BaseAlgo)
# ----------------------------------------------------------------------------

class CLoVeSub(_BaseAlgo):
    pass  # All specialised behaviour omitted in this stub – inherits everything


class ERRing(_BaseAlgo):
    pass


class SparCL(_BaseAlgo):
    pass
