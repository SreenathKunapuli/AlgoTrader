"""Cross-sectional momentum book — code-reviewed constants, like tiers.py.

Provenance: research/scripts/run_xsec.py with the liquidity-at-the-time
screen (eligible_top=300 ≈ top 60% of the ~500-name panel, matching
liquidity_keep=0.60 here). Concentration ladder measured 2026-07-10
(data/xsec_results/profiles.csv, walk-forward OOS 2020–2026.5, net of
10bps, vs SPY 15.8% CAGR / 0.83 Sharpe / −33.8% maxDD):

    top-50: 28.4% CAGR / Sharpe 1.04 / maxDD −36.5%   (validated baseline)
    top-20: 47.3% / 1.26 / −40.4%  worst day −15.8%
    top-10: 69.3% / 1.47 / −42.6%  worst day −15.1%

Concentration amplifies the same edge AND the residual survivorship bias
and regime luck — the honest live expectation remains "SPY plus a
single-digit annual edge" at rising volatility, not the backtest CAGR.
The three risk levels are therefore a concentration + allocation ladder,
with the account catastrophe floor (tiers.py max_drawdown_pct) set beyond
each profile's account-level backtest drawdown.

Execution (2026-07-10 redesign): rebalance orders are priced from live IEX
quotes (not the 16-min-delayed SIP daily close), retried over several
passive rounds, then optionally swept with urgent orders that may cross
the spread — the backtest assumes fills AT the close, so paying ~2-5bps of
spread on the sweep is more faithful than not filling at all. Shortfalls
persist as pending targets and are retried next session; a month-end
missed entirely (engine down) is caught up on the next session.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tiers import Tier


@dataclass(frozen=True)
class XsecConfig:
    universe_csv: str = "data/sp500_constituents.csv"
    top_n: int = 20
    alloc_pct: float = 0.50          # of equity dedicated to this book
    max_position_pct: float = 0.06   # per-position cap (EW weight + drift headroom)
    liquidity_keep: float = 0.60     # keep top 60% by trailing dollar volume
    lookback_days: int = 252         # momentum window (trading days)
    skip_days: int = 21              # skip most recent month (12-1 momentum)
    history_days: int = 420          # calendar days of daily bars to fetch
    rebalance_before_close_min: int = 30  # start window: close-30min .. entry cutoff
    maker_rounds: int = 3            # passive re-quote rounds before the sweep
    market_fallback: bool = True     # final sweep may cross the spread
    catch_up_missed: bool = True     # missed month-end -> rebalance next session
    book_dd_halt_pct: float = 0.40   # within-cycle book DD beyond any backtest
    #                                  month -> halt NEW BUYS (sells still allowed)


# MEDIUM default; also the standalone-import default (backwards compatible).
XSEC = XsecConfig()

# One profile per risk tier. Equal weights are alloc/top_n; the per-position
# cap is that weight plus drift headroom.
XSEC_BY_TIER: dict[Tier, XsecConfig] = {
    Tier.LOW: XsecConfig(top_n=50, alloc_pct=0.50, max_position_pct=0.025),
    Tier.MEDIUM: XSEC,
    Tier.HIGH: XsecConfig(top_n=10, alloc_pct=0.65, max_position_pct=0.13),
}
