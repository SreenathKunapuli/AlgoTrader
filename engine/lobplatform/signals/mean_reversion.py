"""Mean reversion: fade the last-5-min return vs a rolling 2-hour window.

score = −clip(z/3, −1, 1). Confidence decays to 0.2x when the realized-vol
percentile exceeds 90 — don't fade a breakout.
"""

from __future__ import annotations

import numpy as np

from ..data.bar_builder import Bar
from .base import Signal, SignalOutput

WINDOW_BARS = 24  # 2 hours of 5-min bars


class MeanReversionSignal(Signal):
    name = "mean_reversion"

    def compute(self, symbol: str, bars: list[Bar]) -> SignalOutput:
        if len(bars) < WINDOW_BARS + 2:
            return SignalOutput(0.0, 0.0)
        closes = np.array([b.close for b in bars])
        rets = np.diff(np.log(np.maximum(closes, 1e-9)))
        window = rets[-WINDOW_BARS:]
        sd = window.std()
        if sd < 1e-12:
            return SignalOutput(0.0, 0.0)
        z = (rets[-1] - window.mean()) / sd
        score = float(np.clip(-z / 3.0, -1.0, 1.0))
        conf = float(min(abs(z) / 3.0, 1.0))
        # breakout guard: high realized-vol percentile => shrink confidence
        rv = np.sqrt(np.convolve(rets**2, np.ones(20) / 20, mode="valid"))
        if len(rv) > 10:
            pct = float((rv <= rv[-1]).mean())
            if pct > 0.90:
                conf *= 0.2
        return SignalOutput(score, conf)
