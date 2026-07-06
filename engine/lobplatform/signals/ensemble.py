"""Ensemble: tier-weighted, confidence-scaled sum, vol-regime multiplied.

final = Σ(tier_weight_i × health_mult_i × score_i × confidence_i) × vol_mult
Candidates = symbols with |final| ≥ tier confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config.tiers import TierConfig
from ..data.bar_builder import Bar
from .base import Signal
from .vol_regime import vol_multiplier


@dataclass
class EnsembleResult:
    symbol: str
    final_score: float
    vol_mult: float
    per_signal: dict[str, dict[str, float]] = field(default_factory=dict)

    def is_candidate(self, tier: TierConfig) -> bool:
        return abs(self.final_score) >= tier.confidence_threshold


class Ensemble:
    def __init__(self, signals: list[Signal]) -> None:
        self.signals = {s.name: s for s in signals}
        self.health_multipliers: dict[str, float] = {}  # shadow-eval overrides

    def compute(self, symbol: str, bars: list[Bar], tier: TierConfig) -> EnsembleResult:
        total = 0.0
        per: dict[str, dict[str, float]] = {}
        for name, sig in self.signals.items():
            weight = tier.signal_weights.get(name, 0.0)
            out = sig.compute(symbol, bars)
            hm = self.health_multipliers.get(name, 1.0)
            contrib = weight * hm * out.score * out.confidence
            total += contrib
            per[name] = {"score": out.score, "confidence": out.confidence,
                         "weight": weight, "contribution": contrib}
        # vol multiplier is applied to POSITION SIZE (sizing.py), not the score
        vm = vol_multiplier(bars)
        return EnsembleResult(symbol=symbol, final_score=max(-1.0, min(1.0, total)),
                              vol_mult=vm, per_signal=per)
