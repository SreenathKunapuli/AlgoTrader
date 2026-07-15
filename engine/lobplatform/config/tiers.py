"""Risk tier definitions — the exact parameter table from the spec.

Why: tiers are frozen dataclasses (not DB rows, not env vars) so that risk
limits are code-reviewed constants; changing a limit is a diff, not a tweak.

Scope split (2026-07-10): a single account-wide kill switch calibrated for
the intraday book was structurally incompatible with the xsec book (50% of
equity in a ~28%-vol strategy whose own backtest max DD is −37%: normal
momentum drawdowns would market-liquidate the monthly book at the bottom).
Now:
  - daily_loss_limit_pct   -> INTRADAY book day PnL only; breach flattens
                              and halts the intraday book for the day.
  - max_drawdown_pct       -> ACCOUNT catastrophe floor (peak-to-trough,
                              both books), set BEYOND the deployed xsec
                              profile's backtest DD at its allocation
                              (data/xsec_results/profiles.csv):
                              top-50@50% ≈ −18%, top-20@50% ≈ −20%,
                              top-10@65% ≈ −28% account-level. A breach
                              means the strategy is outside anything the
                              backtest ever produced — full kill is right.
  - data staleness         -> intraday scope only (xsec decisions are
                              monthly, marks come via REST, not the stream).
The xsec book additionally halts NEW BUYS when its within-cycle drawdown
exceeds config/xsec.py book_dd_halt_pct — de-risk, never panic-liquidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Tier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_LOW_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "GLD", "TLT"]
_MED_UNIVERSE = _LOW_UNIVERSE + [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "UNH",
]


@dataclass(frozen=True)
class TierConfig:
    name: Tier
    universe: list[str] = field(hash=False)
    allow_short: bool
    max_position_pct: float          # of equity, per position (intraday book)
    max_gross_pct: float             # gross exposure cap (intraday book)
    max_open_positions: int
    daily_loss_limit_pct: float      # intraday-book day loss -> intraday halt
    max_drawdown_pct: float          # ACCOUNT floor, peak equity -> global kill
    stop_atr_multiple: float         # per-position stop = mult * ATR(14, 5m)
    confidence_threshold: float      # ensemble |score| gate
    lob_flow_gate: float             # veto threshold: block long when lob_score < -gate
    #                                  (and short when lob_score > gate); one-sided guard
    lob_flow_min_conf: float         # minimum lob_flow confidence to apply the gate;
    #                                  below this the score is noise — gate is skipped
    rebalance_seconds: int           # cadence (LOW uses daily-at-15:45 handling)
    risk_per_trade_pct: float        # sizing input
    signal_weights: dict[str, float] = field(hash=False)


# Confidence thresholds recalibrated 2026-07-09: three live days (720 ensemble
# computes) never exceeded |score| 0.434 against gates of 0.52-0.60, so the
# engine could not trade at all. The old gates assumed contributions near the
# weight ceiling from all three signals at once; confidence multiplication and
# momentum/mean-reversion self-cancellation (fixed by regime blending in
# ensemble.py) make that unreachable. New gates sit where only a dominant
# regime-blended signal plus corroboration can reach them.
TIERS: dict[Tier, TierConfig] = {
    Tier.LOW: TierConfig(
        name=Tier.LOW, universe=_LOW_UNIVERSE, allow_short=False,
        max_position_pct=0.05, max_gross_pct=0.40, max_open_positions=6,
        daily_loss_limit_pct=0.01, max_drawdown_pct=0.25, stop_atr_multiple=1.5,
        confidence_threshold=0.55, lob_flow_gate=0.10, lob_flow_min_conf=0.30,
        rebalance_seconds=86400, risk_per_trade_pct=0.0025,
        signal_weights={"momentum": 0.7, "mean_reversion": 0.3, "lob_flow": 0.0},
    ),
    Tier.MEDIUM: TierConfig(
        name=Tier.MEDIUM, universe=_MED_UNIVERSE, allow_short=False,
        max_position_pct=0.10, max_gross_pct=0.80, max_open_positions=10,
        daily_loss_limit_pct=0.02, max_drawdown_pct=0.35, stop_atr_multiple=2.0,
        confidence_threshold=0.45, lob_flow_gate=0.10, lob_flow_min_conf=0.30,
        rebalance_seconds=900, risk_per_trade_pct=0.005,
        signal_weights={"momentum": 0.4, "mean_reversion": 0.3, "lob_flow": 0.3},
    ),
    Tier.HIGH: TierConfig(
        name=Tier.HIGH, universe=_MED_UNIVERSE, allow_short=True,
        max_position_pct=0.20, max_gross_pct=1.50, max_open_positions=15,
        daily_loss_limit_pct=0.04, max_drawdown_pct=0.45, stop_atr_multiple=2.5,
        confidence_threshold=0.40, lob_flow_gate=0.10, lob_flow_min_conf=0.30,
        rebalance_seconds=300, risk_per_trade_pct=0.01,
        signal_weights={"momentum": 0.3, "mean_reversion": 0.3, "lob_flow": 0.4},
    ),
}
