"""Time-series momentum on daily closes derived from intraday bars.

Score = sign(20-day return), magnitude = |z| of that return vs its own
1-year (or available) history, clipped to 1. Confidence = min(|z|/2, 1).
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date

import numpy as np

from ..data.bar_builder import Bar
from .base import Signal, SignalOutput


def daily_closes(bars: list[Bar]) -> np.ndarray:
    by_day: OrderedDict[date, float] = OrderedDict()
    for b in bars:
        by_day[b.ts.date()] = b.close  # last bar of the day wins
    return np.array(list(by_day.values()))


class MomentumSignal(Signal):
    name = "momentum"

    def compute(self, symbol: str, bars: list[Bar]) -> SignalOutput:
        closes = daily_closes(bars)
        if len(closes) < 25:
            return SignalOutput(0.0, 0.0)
        rets20 = closes[20:] / closes[:-20] - 1.0
        cur = rets20[-1]
        hist = rets20[:-1]
        sd = hist.std()
        if sd < 1e-12:
            return SignalOutput(0.0, 0.0)
        z = (cur - hist.mean()) / sd
        score = float(np.clip(np.sign(cur) * min(abs(z) / 3.0, 1.0), -1.0, 1.0))
        return SignalOutput(score, float(min(abs(z) / 2.0, 1.0)))
