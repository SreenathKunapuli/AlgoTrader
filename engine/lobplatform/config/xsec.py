"""Cross-sectional momentum book — code-reviewed constants, like tiers.py.

Provenance: research/scripts/run_xsec.py + run_xsec2.py (2026-07-09).
Deployed exactly what survived the tournament: large-cap 12-1 momentum,
liquidity-filtered, monthly, long-only. TOP_N=20 (not the backtest's 50)
because a $100k paper book slices into untradeably small lots at 50 names;
concentration raises tracking error, which the user accepted explicitly
("high risk reward" — concentration on a positive edge, not frequency).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XsecConfig:
    universe_csv: str = "data/sp500_constituents.csv"
    top_n: int = 20
    alloc_pct: float = 0.50          # of equity dedicated to this book
    max_position_pct: float = 0.06   # per-position cap (2.5% equal-weight + drift headroom)
    liquidity_keep: float = 0.60     # keep top 60% by trailing dollar volume
    lookback_days: int = 252         # momentum window (trading days)
    skip_days: int = 21              # skip most recent month (12-1 momentum)
    history_days: int = 420          # calendar days of daily bars to fetch
    rebalance_before_close_min: int = 20  # start window: close-20min .. entry cutoff


XSEC = XsecConfig()
