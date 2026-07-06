"""Loss functions for imbalanced 3-class LOB prediction.

With realistic label thresholds, FLAT dominates; plain cross-entropy yields
a model that calls everything flat and posts a deceptively high accuracy.
Focal loss (Lin et al. 2017) down-weights easy examples via the (1-p)^gamma
factor and accepts per-class alpha weights on top.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(labels: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency weights normalized to mean 1 (train labels only)."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
