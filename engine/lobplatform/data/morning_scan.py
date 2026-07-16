"""Pre-market daily momentum scan over the broad universe.

Loads the broadest available universe CSV (universe3000.csv → sp500_constituents.csv),
fetches 60 days of daily bars via REST (chunked, no streaming limit), scores each stock
by short-term momentum, and returns the top N candidates for the day's streaming
subscription — before the first live minute bar arrives.

Scoring (daily bars only, no LOB):
  60% 20-day momentum  — trend continuation; proven winners keep moving
  40%  5-day momentum  — very recent strength; catches stocks just starting to break out

Only positive-momentum stocks are returned. Both weights must be positive for a stock
to appear — this prevents a stock with a great 20-day run but a bad recent week from
crowding out a fresh breakout name.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

UNIVERSE_PATHS = [
    "data/universe3000.csv",
    "data/sp500_constituents.csv",
]


def load_universe(extra_paths: list[str] | None = None) -> list[str]:
    """Return sorted symbols from the broadest available universe CSV.

    Supports both 'symbol' (universe3000.csv) and 'Symbol' (sp500_constituents.csv)
    column headers. Falls back gracefully if neither file exists.
    """
    for p in (extra_paths or []) + UNIVERSE_PATHS:
        path = Path(p)
        if not path.exists():
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            symbols = []
            for row in reader:
                sym = (row.get("symbol") or row.get("Symbol") or "").strip()
                if sym and sym.isalpha() and 1 <= len(sym) <= 5:
                    symbols.append(sym)
        if symbols:
            log.info("morning_scan.universe", path=str(path), n=len(symbols))
            return sorted(set(symbols))
    log.warning("morning_scan.no_universe_file", searched=UNIVERSE_PATHS)
    return []


def score_universe(
    history: dict[str, dict[str, Any]],
    min_price: float = 5.0,
    min_dollar_vol: float = 3_000_000.0,
) -> list[tuple[str, float]]:
    """Score every symbol in `history` and return (symbol, score) sorted best-first.

    Requires >= 22 daily closes (1 month of trading data). Filters on price and
    trailing dollar volume so the output is always tradeable on the Basic plan.
    Only stocks with BOTH positive 20-day and positive 5-day momentum are included —
    this keeps the list focused on confirmed trends, not mean-reversion bets.
    """
    scored: list[tuple[str, float]] = []
    for sym, data in history.items():
        closes = data.get("closes", [])
        dv = data.get("dollar_vol", 0.0)
        if len(closes) < 22 or closes[-1] < min_price or dv < min_dollar_vol:
            continue
        mom_20d = closes[-1] / closes[-22] - 1.0
        mom_5d = closes[-1] / closes[-6] - 1.0
        if mom_20d <= 0 or mom_5d <= 0:
            continue
        score = 0.6 * mom_20d + 0.4 * mom_5d
        scored.append((sym, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def build_daily_candidates(
    api_key: str,
    secret_key: str,
    top_n: int = 22,
    exclude: set[str] | None = None,
    history_days: int = 60,
) -> list[str]:
    """Score the full broad universe and return up to top_n momentum candidates.

    Fetches daily bars for all universe symbols (chunked REST, typically 3-15 API
    calls depending on universe size). Excludes symbols already in the live universe
    or currently held by the xsec book so we don't duplicate subscriptions.

    Returns symbols sorted best-first by momentum score.
    """
    from .history import fetch_daily_history

    all_symbols = load_universe()
    if not all_symbols:
        return []

    exc = exclude or set()
    symbols = [s for s in all_symbols if s not in exc]

    log.info("morning_scan.fetch_start", universe=len(symbols), days=history_days)
    try:
        history = fetch_daily_history(api_key, secret_key, symbols, days=history_days)
    except Exception as e:
        log.error("morning_scan.fetch_failed", error=str(e))
        return []

    ranked = score_universe(history)
    picks = [sym for sym, _score in ranked if sym not in exc][:top_n]
    log.info("morning_scan.ranked",
             scored=len(ranked), selected=len(picks), top5=picks[:5])
    return picks
