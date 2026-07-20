"""REST backfill of historical 1-min bars for signal warmup.

Why: signals need lookback (momentum needs ~1y of daily closes derived from
minute bars; the NN needs 64 five-minute bars) before the first live bar.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .bar_builder import Bar


def fetch_daily_history(
    api_key: str, secret_key: str, symbols: list[str], days: int = 420
) -> dict[str, dict[str, Any]]:
    """Daily closes + trailing dollar volume for the xsec book.

    adjustment="all" folds splits and dividends into close (matches the
    research backtest exactly); feed="sip" because IEX returns nothing for
    daily history on this account (verified 2026-07-09).
    Returns {symbol: {"closes": [float], "dollar_vol": float}}.
    """
    client = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.now(UTC) - timedelta(minutes=16)
    start = end - timedelta(days=days)
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i: i + 200]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                               start=start, end=end, adjustment=Adjustment.ALL, feed=DataFeed.SIP)
        resp: Any = client.get_stock_bars(req)
        for sym in chunk:
            bars = resp.data.get(sym, [])
            if not bars:
                continue
            closes = [float(b.close) for b in bars]
            dv = [float(b.close) * float(b.volume) for b in bars[-63:]]
            out[sym] = {"closes": closes, "dollar_vol": sum(dv) / len(dv) if dv else 0.0}
    return out


def fetch_latest_closes(
    api_key: str, secret_key: str, symbols: list[str]
) -> dict[str, float]:
    """Most recent daily close per symbol (mark refresh for xsec holdings)."""
    if not symbols:
        return {}
    client = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.now(UTC) - timedelta(minutes=16)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                           start=end - timedelta(days=7), end=end,
                           adjustment=Adjustment.ALL, feed=DataFeed.SIP)
    resp: Any = client.get_stock_bars(req)
    return {s: float(resp.data[s][-1].close) for s in symbols if resp.data.get(s)}


def fetch_latest_quotes(
    api_key: str, secret_key: str, symbols: list[str]
) -> dict[str, tuple[float, float]]:
    """{symbol: (bid, ask)} from the free REAL-TIME IEX feed.

    Why: SIP history on the free tier is 15-min delayed — pricing rebalance
    limit orders off it systematically misses names that moved (the ones
    momentum just picked). IEX latest quotes are real-time; thin names may
    come back zero/crossed and are simply omitted (caller falls back to the
    daily-close mark).
    """
    if not symbols:
        return {}
    from alpaca.data.requests import StockLatestQuoteRequest

    client = StockHistoricalDataClient(api_key, secret_key)
    out: dict[str, tuple[float, float]] = {}
    for i in range(0, len(symbols), 200):
        chunk = symbols[i: i + 200]
        req = StockLatestQuoteRequest(symbol_or_symbols=chunk, feed=DataFeed.IEX)
        quotes: Any = client.get_stock_latest_quote(req)
        for sym in chunk:
            q = quotes.get(sym)
            if q is None:
                continue
            bid, ask = float(q.bid_price or 0.0), float(q.ask_price or 0.0)
            if bid > 0.0 and ask > bid:
                out[sym] = (bid, ask)
    return out


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
