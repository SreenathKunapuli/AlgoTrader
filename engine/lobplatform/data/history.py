"""REST backfill of historical 1-min bars for signal warmup.

Why: signals need lookback (momentum needs ~1y of daily closes derived from
minute bars; the NN needs 64 five-minute bars) before the first live bar.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .bar_builder import Bar


def fetch_minute_bars(
    api_key: str, secret_key: str, symbols: list[str], days: int = 30
) -> dict[str, list[Bar]]:
    """Fetch `days` of 1-min bars per symbol via Alpaca data REST (IEX feed)."""
    client = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.now(UTC) - timedelta(minutes=16)  # free-tier delay margin
    start = end - timedelta(days=days * 1.5)  # calendar padding for weekends/holidays
    req = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Minute, start=start, end=end
    )
    resp: Any = client.get_stock_bars(req)
    out: dict[str, list[Bar]] = {s: [] for s in symbols}
    for sym in symbols:
        for b in resp.data.get(sym, []):
            out[sym].append(
                Bar(
                    symbol=sym, ts=b.timestamp.astimezone(UTC), interval_s=60,
                    open=float(b.open), high=float(b.high), low=float(b.low),
                    close=float(b.close), volume=int(b.volume),
                    vwap=float(b.vwap or b.close), trade_count=int(b.trade_count or 0),
                    mean_spread=0.0, mean_quote_imbalance=0.0, flow_imbalance=0.0,
                )
            )
    return out
