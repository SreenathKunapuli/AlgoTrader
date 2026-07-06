"""Label construction for mid-price direction prediction.

Implements the smooth-label scheme from DeepLOB (Zhang et al. 2019):

    m+(t) = mean(mid[t+1 .. t+k])          # smoothed future mid
    l(t)  = (m+(t) - mid[t]) / mid[t]      # pct change vs current mid

    label = UP   if l >  alpha
            DOWN if l < -alpha
            FLAT otherwise

Smoothing over k steps removes bid-ask bounce noise from the target. The
last k samples have no complete future window and are marked invalid (-1);
training code must drop them — silently labelling them is lookahead bias.

Class encoding: 0 = DOWN, 1 = FLAT, 2 = UP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DOWN, FLAT, UP = 0, 1, 2
INVALID = -1


@dataclass
class LabelConfig:
    horizon: int = 20          # k: smoothing/prediction window in snapshots
    alpha: float = 5e-5        # threshold on pct mid change
    mode: str = "vs_current"   # "vs_current" (m+ vs mid_t) or "vs_past" (m+ vs m-)


def make_labels(mid: np.ndarray, cfg: LabelConfig) -> np.ndarray:
    """Return int labels [N]; trailing/leading incomplete windows are INVALID."""
    n = len(mid)
    k = cfg.horizon
    if n <= 2 * k:
        raise ValueError(f"series too short for horizon {k}")

    # forward mean: m_plus[t] = mean(mid[t+1 .. t+k])
    csum = np.cumsum(np.concatenate([[0.0], mid]))
    m_plus = np.full(n, np.nan)
    m_plus[: n - k] = (csum[k + 1 :] - csum[1 : n - k + 1]) / k

    if cfg.mode == "vs_current":
        ref = mid
        valid_from = 0
    elif cfg.mode == "vs_past":
        # m_minus[t] = mean(mid[t-k+1 .. t])
        m_minus = np.full(n, np.nan)
        m_minus[k - 1 :] = (csum[k:] - csum[: n - k + 1]) / k
        ref = m_minus
        valid_from = k - 1
    else:
        raise ValueError(f"unknown label mode: {cfg.mode}")

    l = (m_plus - ref) / ref

    labels = np.full(n, INVALID, dtype=np.int64)
    valid = slice(valid_from, n - k)
    lv = l[valid]
    lab = np.full(lv.shape, FLAT, dtype=np.int64)
    lab[lv > cfg.alpha] = UP
    lab[lv < -cfg.alpha] = DOWN
    labels[valid] = lab
    return labels


def class_distribution(labels: np.ndarray) -> dict[str, float]:
    valid = labels[labels != INVALID]
    n = len(valid)
    return {
        "down": float((valid == DOWN).sum()) / n,
        "flat": float((valid == FLAT).sum()) / n,
        "up": float((valid == UP).sum()) / n,
        "n_valid": n,
    }


def suggest_alpha(
    mid: np.ndarray, horizon: int, target_flat: float = 0.4
) -> float:
    """Pick alpha so that ~target_flat of samples are FLAT.

    Tune this on the TRAINING slice only. The DeepLOB-paper trap: a tighter
    alpha gives a 'harder' 3-class problem with huge class imbalance; a wider
    alpha makes accuracy look great while the signal is untradeable.
    """
    k = horizon
    csum = np.cumsum(np.concatenate([[0.0], mid]))
    n = len(mid)
    m_plus = (csum[k + 1 :] - csum[1 : n - k + 1]) / k
    l = np.abs((m_plus - mid[: n - k]) / mid[: n - k])
    return float(np.quantile(l, target_flat))
