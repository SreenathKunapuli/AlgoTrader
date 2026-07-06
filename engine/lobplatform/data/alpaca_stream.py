"""Async Alpaca IEX stream consumer with reconnect + heartbeat.

Why: the staleness monitor (risk layer) keys off `last_tick_ts`; the stream
must never die silently, so reconnects use exponential backoff (1s -> 60s
cap) and always resubscribe the full universe.
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


class MarketStream:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbols: list[str],
        on_trade: TradeHandler,
        on_quote: QuoteHandler,
    ) -> None:
        self.symbols = symbols
        self.on_trade = on_trade
        self.on_quote = on_quote
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

    async def run_forever(self) -> None:
        """Connect, stream, reconnect with backoff until stop() is called."""
        from alpaca.data.live import StockDataStream

        backoff = 1.0
        while not self._stop.is_set():
            try:
                stream = StockDataStream(self._api_key, self._secret_key)
                stream.subscribe_trades(self._handle_trade, *self.symbols)
                stream.subscribe_quotes(self._handle_quote, *self.symbols)
                log.info("stream.connect", symbols=len(self.symbols))
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
