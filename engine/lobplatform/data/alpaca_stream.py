"""Async Alpaca IEX stream consumer with reconnect + heartbeat.

Free-tier reality: the Basic data plan allows 30 symbol subscriptions per
connection. Strategy (see DECISIONS.md):
  1. subscribe the official 1-MIN BAR channel for every symbol (N subs) —
     server-built OHLCV, no local bar assembly needed on the live path;
  2. spend the remaining budget on QUOTES (spread/imbalance features),
     then TRADES (signed flow) for as many leading symbols as fit.
Symbols without quote/trade subs get zeros for those microstructure
features — the models tolerate missing features far better than the
engine tolerates a dead feed (which is what over-subscribing causes).

Reconnects use exponential backoff (1s -> 60s cap) and resubscribe.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog

log = structlog.get_logger()

TradeHandler = Callable[[str, datetime, float, int], Awaitable[None]]
QuoteHandler = Callable[[str, datetime, float, int, float, int], Awaitable[None]]
# symbol, bar_open_ts, o, h, l, c, volume, vwap, trade_count
BarHandler = Callable[[str, datetime, float, float, float, float, int, float, int],
                      Awaitable[None]]

SUBSCRIPTION_LIMIT = 30


def plan_subscriptions(symbols: list[str], limit: int = SUBSCRIPTION_LIMIT
                       ) -> tuple[list[str], list[str], list[str]]:
    """Allocate the budget: bars for all, then quotes, then trades."""
    n = len(symbols)
    if n > limit:
        symbols = symbols[:limit]
        n = limit
    remaining = limit - n
    quotes = symbols[: min(n, remaining)]
    remaining -= len(quotes)
    trades = symbols[: min(n, remaining)]
    return symbols, quotes, trades


class MarketStream:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbols: list[str],
        on_trade: TradeHandler,
        on_quote: QuoteHandler,
        on_bar: BarHandler,
    ) -> None:
        self.symbols = symbols
        self.on_trade = on_trade
        self.on_quote = on_quote
        self.on_bar = on_bar
        self.last_tick_ts: datetime = datetime.now(UTC)
        self._api_key = api_key
        self._secret_key = secret_key
        self._stop = asyncio.Event()

    async def _handle_trade(self, t: Any) -> None:
        self.last_tick_ts = datetime.now(UTC)
        await self.on_trade(t.symbol, t.timestamp.astimezone(UTC),
                            float(t.price), int(t.size))

    async def _handle_quote(self, q: Any) -> None:
        self.last_tick_ts = datetime.now(UTC)
        await self.on_quote(q.symbol, q.timestamp.astimezone(UTC),
                            float(q.bid_price), int(q.bid_size),
                            float(q.ask_price), int(q.ask_size))

    async def _handle_bar(self, b: Any) -> None:
        self.last_tick_ts = datetime.now(UTC)
        await self.on_bar(b.symbol, b.timestamp.astimezone(UTC),
                          float(b.open), float(b.high), float(b.low),
                          float(b.close), int(b.volume),
                          float(b.vwap or b.close), int(b.trade_count or 0))

    async def run_forever(self) -> None:
        """Connect, stream, reconnect with backoff until stop() is called."""
        from alpaca.data.live import StockDataStream

        bar_syms, quote_syms, trade_syms = plan_subscriptions(self.symbols)
        backoff = 1.0
        while not self._stop.is_set():
            try:
                stream = StockDataStream(self._api_key, self._secret_key)
                stream.subscribe_bars(self._handle_bar, *bar_syms)
                if quote_syms:
                    stream.subscribe_quotes(self._handle_quote, *quote_syms)
                if trade_syms:
                    stream.subscribe_trades(self._handle_trade, *trade_syms)
                log.info("stream.connect", bars=len(bar_syms),
                         quotes=len(quote_syms), trades=len(trade_syms))
                backoff = 1.0
                await stream._run_forever()  # alpaca-py's internal async runner
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("stream.disconnect", error=str(exc), retry_in=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def stop(self) -> None:
        self._stop.set()
