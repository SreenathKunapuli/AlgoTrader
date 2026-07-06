"""Tick -> bar aggregation with per-bar microstructure features.

Why: signals may only see *finalized* bars (leakage rule). The builder holds
one accumulator per (symbol, interval) and emits a Bar only when the wall
clock crosses the interval boundary; partial state is never exposed.

Trade signing uses the quote rule: a trade at/above the prevailing ask is a
buy (+), at/below the bid a sell (−), else sign of price change (tick rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: datetime            # bar OPEN time, UTC; finalized at ts + interval
    interval_s: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int
    mean_spread: float      # mean bid-ask spread over the bar
    mean_quote_imbalance: float  # mean (bid_sz - ask_sz)/(bid_sz + ask_sz)
    flow_imbalance: float   # (buy_vol - sell_vol)/(buy_vol + sell_vol)


@dataclass
class _Acc:
    open: float = 0.0
    high: float = -1e30
    low: float = 1e30
    close: float = 0.0
    volume: int = 0
    notional: float = 0.0
    trade_count: int = 0
    spread_sum: float = 0.0
    spread_n: int = 0
    qimb_sum: float = 0.0
    qimb_n: int = 0
    buy_vol: int = 0
    sell_vol: int = 0
    started: bool = False


def bar_start(ts: datetime, interval_s: int) -> datetime:
    epoch = int(ts.timestamp())
    return datetime.fromtimestamp(epoch - epoch % interval_s, tz=UTC)


class BarBuilder:
    """Feed ticks; collect finalized bars from on_trade/on_quote/flush returns."""

    def __init__(self, interval_s: int = 60) -> None:
        self.interval_s = interval_s
        self._acc: dict[str, _Acc] = {}
        self._bar_ts: dict[str, datetime] = {}
        self._last_bid: dict[str, tuple[float, int]] = {}
        self._last_ask: dict[str, tuple[float, int]] = {}
        self._last_price: dict[str, float] = {}

    def _roll(self, symbol: str, ts: datetime) -> Bar | None:
        """Finalize the open bar if `ts` has crossed into a new interval."""
        start = bar_start(ts, self.interval_s)
        prev = self._bar_ts.get(symbol)
        out: Bar | None = None
        if prev is not None and start > prev:
            out = self._finalize(symbol, prev)
        if prev is None or start > prev:
            self._bar_ts[symbol] = start
            self._acc[symbol] = _Acc()
        return out

    def _finalize(self, symbol: str, ts: datetime) -> Bar | None:
        a = self._acc.get(symbol)
        if a is None or not a.started:
            return None
        tot = a.buy_vol + a.sell_vol
        return Bar(
            symbol=symbol, ts=ts, interval_s=self.interval_s,
            open=a.open, high=a.high, low=a.low, close=a.close,
            volume=a.volume,
            vwap=a.notional / a.volume if a.volume else a.close,
            trade_count=a.trade_count,
            mean_spread=a.spread_sum / a.spread_n if a.spread_n else 0.0,
            mean_quote_imbalance=a.qimb_sum / a.qimb_n if a.qimb_n else 0.0,
            flow_imbalance=(a.buy_vol - a.sell_vol) / tot if tot else 0.0,
        )

    def on_quote(self, symbol: str, ts: datetime, bid: float, bid_size: int,
                 ask: float, ask_size: int) -> Bar | None:
        done = self._roll(symbol, ts)
        self._last_bid[symbol] = (bid, bid_size)
        self._last_ask[symbol] = (ask, ask_size)
        a = self._acc[symbol]
        if ask > 0 and bid > 0:
            a.spread_sum += ask - bid
            a.spread_n += 1
        denom = bid_size + ask_size
        if denom > 0:
            a.qimb_sum += (bid_size - ask_size) / denom
            a.qimb_n += 1
        return done

    def on_trade(self, symbol: str, ts: datetime, price: float, size: int) -> Bar | None:
        done = self._roll(symbol, ts)
        a = self._acc[symbol]
        if not a.started:
            a.open = price
            a.started = True
        a.high = max(a.high, price)
        a.low = min(a.low, price)
        a.close = price
        a.volume += size
        a.notional += price * size
        a.trade_count += 1
        # quote-rule signing with tick-rule fallback
        bid = self._last_bid.get(symbol, (0.0, 0))[0]
        ask = self._last_ask.get(symbol, (0.0, 0))[0]
        last = self._last_price.get(symbol)
        if ask > 0 and price >= ask:
            a.buy_vol += size
        elif bid > 0 and price <= bid:
            a.sell_vol += size
        elif last is not None and price != last:
            if price > last:
                a.buy_vol += size
            else:
                a.sell_vol += size
        self._last_price[symbol] = price
        return done

    def flush(self, symbol: str, now: datetime) -> Bar | None:
        """Finalize on clock tick (no trades needed to close a bar)."""
        return self._roll(symbol, now)


def aggregate(bars: list[Bar], interval_s: int) -> Bar | None:
    """Aggregate consecutive finalized bars (e.g. 5x1min -> 1x5min)."""
    if not bars:
        return None
    vol = sum(b.volume for b in bars)
    tot_notional = sum(b.vwap * b.volume for b in bars)
    spread_n = sum(1 for b in bars if b.mean_spread > 0)
    qimb_n = sum(1 for b in bars if b.mean_quote_imbalance != 0.0)
    signed = sum(b.flow_imbalance * b.volume for b in bars)
    return Bar(
        symbol=bars[0].symbol, ts=bars[0].ts, interval_s=interval_s,
        open=bars[0].open,
        high=max(b.high for b in bars),
        low=min(b.low for b in bars),
        close=bars[-1].close,
        volume=vol,
        vwap=tot_notional / vol if vol else bars[-1].close,
        trade_count=sum(b.trade_count for b in bars),
        mean_spread=sum(b.mean_spread for b in bars) / max(spread_n, 1),
        mean_quote_imbalance=sum(b.mean_quote_imbalance for b in bars) / max(qimb_n, 1),
        flow_imbalance=signed / vol if vol else 0.0,
    )
