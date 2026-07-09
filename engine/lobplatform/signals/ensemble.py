"""Ensemble: tier-weighted, confidence-scaled sum, vol-regime multiplied.

final = Σ(eff_weight_i × health_mult_i × score_i × confidence_i) × vol_mult
Candidates = symbols with |final| ≥ tier confidence threshold.

Regime blending: momentum and mean-reversion are opposing bets on the same
tape (observed live: opposite sign ~44% of the time both were active, so
their static-weight sum self-cancelled and the ensemble never reached any
tier gate). Their POOLED tier weight is reallocated each compute by
Kaufman's efficiency ratio over mean-reversion's own 2-hour window:
trending tape → momentum gets the pool, choppy tape → mean-reversion does.
lob_flow's weight is untouched — it is not part of the opposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config.tiers import TierConfig
from ..data.bar_builder import Bar
from .base import Signal
from .mean_reversion import WINDOW_BARS
from .vol_regime import vol_multiplier

# ER of a pure random walk over n bars ~ 1/sqrt(n) ≈ 0.20 for n=24; map
# [0.10, 0.35] linearly onto [all-MR, all-momentum] so a random-walk tape
# lands mid-blend rather than at either pole.
ER_FLOOR = 0.10
ER_CEIL = 0.35


def trend_weight(bars: list[Bar], n: int = WINDOW_BARS) -> float:
    """[0, 1]: 0 = choppy (favor mean-reversion), 1 = trending (favor momentum)."""
    closes = np.array([b.close for b in bars[-(n + 1):]])
    if len(closes) < n + 1:
        return 0.5
    net = abs(float(closes[-1] - closes[0]))
    path = float(np.abs(np.diff(closes)).sum())
    if path <= 0:
        return 0.5
    er = net / path
    return float(np.clip((er - ER_FLOOR) / (ER_CEIL - ER_FLOOR), 0.0, 1.0))


@dataclass
class EnsembleResult:
    symbol: str
    final_score: float
    vol_mult: float
    trend_w: float = 0.5
    per_signal: dict[str, dict[str, float]] = field(default_factory=dict)

    def is_candidate(self, tier: TierConfig) -> bool:
        return abs(self.final_score) >= tier.confidence_threshold


class Ensemble:
    def __init__(self, signals: list[Signal]) -> None:
        self.signals = {s.name: s for s in signals}
        self.health_multipliers: dict[str, float] = {}  # shadow-eval overrides

    def _effective_weights(self, tier: TierConfig,
                           bars: list[Bar]) -> tuple[dict[str, float], float]:
        weights = dict(tier.signal_weights)
        tw = trend_weight(bars)
        pool = weights.get("momentum", 0.0) + weights.get("mean_reversion", 0.0)
        if pool > 0 and "momentum" in weights and "mean_reversion" in weights:
            weights["momentum"] = pool * tw
            weights["mean_reversion"] = pool * (1.0 - tw)
        return weights, tw

    def compute(self, symbol: str, bars: list[Bar], tier: TierConfig) -> EnsembleResult:
        weights, tw = self._effective_weights(tier, bars)
        total = 0.0
        per: dict[str, dict[str, float]] = {}
        for name, sig in self.signals.items():
            weight = weights.get(name, 0.0)
            out = sig.compute(symbol, bars)
            hm = self.health_multipliers.get(name, 1.0)
            contrib = weight * hm * out.score * out.confidence
            total += contrib
            per[name] = {"score": out.score, "confidence": out.confidence,
                         "weight": weight, "contribution": contrib}
        # vol multiplier is applied to POSITION SIZE (sizing.py), not the score
        vm = vol_multiplier(bars)
        return EnsembleResult(symbol=symbol, final_score=max(-1.0, min(1.0, total)),
                              vol_mult=vm, trend_w=tw, per_signal=per)
