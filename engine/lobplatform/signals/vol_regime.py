"""Volatility regime gate — not directional; a position-size multiplier.

Realized vol (20-bar, annualized) percentile vs 30-day history maps to a
multiplier in [0.3, 1.0]: calm -> 1.0, extreme -> 0.3.
"""

from __future__ import annotations

import numpy as np

from ..data.bar_builder import Bar


def vol_multiplier(bars: list[Bar]) -> float:
    if len(bars) < 40:
        return 1.0
    closes = np.array([b.close for b in bars])
    rets = np.diff(np.log(np.maximum(closes, 1e-9)))
    rv = np.sqrt(np.convolve(rets**2, np.ones(20) / 20, mode="valid"))
    if len(rv) < 10:
        return 1.0
    pct = float((rv <= rv[-1]).mean())
    if pct <= 0.5:
        return 1.0
    # linear 1.0 -> 0.3 as percentile goes 0.5 -> 1.0
    return float(max(0.3, 1.0 - 1.4 * (pct - 0.5)))
