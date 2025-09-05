[UPDATED FILE]
```
"""
train.py – model architectures, buffers and training algorithms
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

# (helpers unchanged ...)
```

Key change – `_BaseAlgo.__init__` now builds a backbone that really returns the 512-d pre-classifier features instead of an empty (0-d) tensor:
```python
self.backbone = tvm.resnet18(weights=None).to(self.device)
# Strip the final linear classification head so that the network outputs the
# 512-dimensional pooled feature vector (avg-pool output)
self.backbone.fc = nn.Identity()
self.feat_dim = 512  # output dimension after removing the classifier
```

Because the ResNet’s `fc` layer is replaced by `Identity`, the forward pass produces a `[B, 512]` tensor and the subsequent projection `x @ W (512×rank)` is valid.  Nothing else in the file needed to change.
