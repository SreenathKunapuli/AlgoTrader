"""Daily bar acquisition for the cross-sectional universe.

Alpaca SIP daily history starts 2016; adjustment="all" folds splits AND
dividends into close, so day-over-day close ratios are total returns.
Universe is the CURRENT S&P 500 constituent list — that is survivorship
biased (winners only), which inflates absolute backtest returns. The
equal-weight-universe benchmark in backtest.py shares the same bias, so
strategy-minus-EW is the honest measure of ranking skill; strategy-vs-SPY
is the headline number and must be read with the bias in mind.

Cache format is pickled DataFrame (no pyarrow in this venv).
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

BATCH = 100  # symbols per request; alpaca-py paginates internally
START = datetime(2016, 1, 1)


def load_universe(constituents_csv: str) -> list[str]:
    df = pd.read_csv(constituents_csv)
    # Alpaca uses dots for share classes as-is (BRK.B); the CSV already does.
    return sorted(df["Symbol"].astype(str).str.strip().unique().tolist())


def fetch_daily(symbols: list[str], cache_path: str,
                end: datetime | None = None) -> pd.DataFrame:
    """Fetch (or load cached) daily bars, long format indexed by (symbol, ts)."""
    cache = Path(cache_path)
    if cache.exists():
        return pd.read_pickle(cache)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    frames: list[pd.DataFrame] = []
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i: i + BATCH]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                               start=START, end=end, adjustment="all", feed="sip")
        df = client.get_stock_bars(req).df
        if len(df):
            frames.append(df[["open", "high", "low", "close", "volume", "vwap"]])
        print(f"fetched {i + len(chunk)}/{len(symbols)} symbols, "
              f"{sum(len(f) for f in frames):,} rows", flush=True)
        time.sleep(1.0)  # stay far under the 200 req/min limit
    out = pd.concat(frames)
    out.index.names = ["symbol", "ts"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cache)
    return out


def to_panels(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long (symbol, ts) frame -> {field: wide DataFrame [date x symbol]}.

    Dates are normalized to date (bars are daily); symbols with fewer than
    252 observations are dropped (need a year of history for features).
    """
    wide: dict[str, pd.DataFrame] = {}
    d = df.reset_index()
    d["date"] = pd.to_datetime(d["ts"]).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    counts = d.groupby("symbol").size()
    keep = counts[counts >= 252].index
    d = d[d["symbol"].isin(keep)]
    for field in ["close", "volume", "vwap"]:
        wide[field] = d.pivot_table(index="date", columns="symbol",
                                    values=field, aggfunc="last").sort_index()
    return wide
