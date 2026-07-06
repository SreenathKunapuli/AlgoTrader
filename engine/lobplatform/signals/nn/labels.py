"""Smooth 3-class labels — research methodology, retargeted to 5-min bars.

l(t) = (mean(close[t+1..t+k]) - close[t]) / close[t]; UP if l > alpha,
DOWN if l < -alpha, else FLAT. Trailing k bars are INVALID (no complete
future window — labeling them would be lookahead). Alpha is calibrated
per symbol on TRAIN data so FLAT lands in [0.40, 0.60].
"""

from __future__ import annotations

import numpy as np

DOWN, FLAT, UP = 0, 1, 2
INVALID = -1
HORIZON_K = 3


def make_labels(close: np.ndarray, k: int, alpha: float) -> np.ndarray:
    n = len(close)
    labels = np.full(n, INVALID, dtype=np.int64)
    if n <= k:
        return labels
    csum = np.cumsum(np.concatenate([[0.0], close]))
    m_plus = (csum[k + 1:] - csum[1: n - k + 1]) / k
    rel = (m_plus - close[: n - k]) / np.maximum(close[: n - k], 1e-9)
    lab = np.full(n - k, FLAT, dtype=np.int64)
    lab[rel > alpha] = UP
    lab[rel < -alpha] = DOWN
    labels[: n - k] = lab
    return labels


def calibrate_alpha(close: np.ndarray, k: int, target_flat: float = 0.5) -> float:
    """Pick alpha so ~target_flat of TRAIN samples are FLAT (quantile of |l|)."""
    n = len(close)
    if n <= k + 10:
        return 1e-4
    csum = np.cumsum(np.concatenate([[0.0], close]))
    m_plus = (csum[k + 1:] - csum[1: n - k + 1]) / k
    rel = np.abs((m_plus - close[: n - k]) / np.maximum(close[: n - k], 1e-9))
    return float(max(np.quantile(rel, target_flat), 1e-6))
