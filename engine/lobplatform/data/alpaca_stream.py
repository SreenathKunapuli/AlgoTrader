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
        self.symbols = list(symbols)
        self.on_trade = on_trade
        self.on_quote = on_quote
        self.on_bar = on_bar
        self.last_tick_ts: datetime = datetime.now(UTC)
        self._api_key = api_key
        self._secret_key = secret_key
        self._stop = asyncio.Event()
        self._reconnect = asyncio.Event()  # set by update_symbols() to trigger reconnect

    def update_symbols(self, new_symbols: list[str]) -> None:
        """Swap in a new symbol list and trigger a stream reconnect.

        The current connection is torn down cleanly; run_forever() immediately
        re-connects with the updated subscription plan.
        """
        self.symbols = list(new_symbols)
        self._reconnect.set()

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
        """Connect, stream, reconnect with backoff until stop() is called.

        update_symbols() sets _reconnect which tears down the current connection
        cleanly (no backoff) and immediately reconnects with the new symbol list.
        """
        from alpaca.data.live import StockDataStream

        backoff = 1.0
        while not self._stop.is_set():
            self._reconnect.clear()
            bar_syms, quote_syms, trade_syms = plan_subscriptions(self.symbols)
            try:
                stream = StockDataStream(self._api_key, self._secret_key)
                stream.subscribe_bars(self._handle_bar, *bar_syms)
                if quote_syms:
                    stream.subscribe_quotes(self._handle_quote, *quote_syms)
                if trade_syms:
                    stream.subscribe_trades(self._handle_trade, *trade_syms)
                log.info("stream.connect", bars=len(bar_syms),
                         quotes=len(quote_syms), trades=len(trade_syms))

                stream_task = asyncio.ensure_future(stream._run_forever())
                reconnect_task = asyncio.ensure_future(self._reconnect.wait())
                done, pending = await asyncio.wait(
                    {stream_task, reconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

                if reconnect_task in done:
                    log.info("stream.reconnecting", symbols=len(self.symbols))
                    backoff = 1.0
                    continue  # immediately reconnect with new self.symbols

                # stream died unexpectedly — raise its exception for backoff
                if stream_task in done and not stream_task.cancelled():
                    exc = stream_task.exception()
                    if exc:
                        raise exc
                backoff = 1.0

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("stream.disconnect", error=str(exc), retry_in=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def stop(self) -> None:
        self._stop.set()


# symbol, side, filled qty (this event), fill price, client_order_id
FillHandler = Callable[[str, str, int, float, str], Awaitable[None]]


class TradeUpdateStream:
    """Trade-update (fill) stream -> real-time position mirror.

    Before this, on_fill had no live caller: the mirror was only trued by
    periodic broker reconciles, so stops and risk caps acted on state up to
    one reconcile-cycle stale. Fill/partial_fill events now update it in
    real time; reconciles remain the backstop source of truth.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool,
                 on_fill: FillHandler) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self.on_fill = on_fill
        self._stop = asyncio.Event()

    async def _handle(self, data: Any) -> None:
        event = str(getattr(data, "event", ""))
        if event not in ("fill", "partial_fill"):
            return
        order = getattr(data, "order", None)
        if order is None:
            return
        try:
            symbol = str(order.symbol)
            side = getattr(order.side, "value", str(order.side)).lower()
            qty = int(float(data.qty or 0))
            price = float(data.price or 0.0)
            coid = str(getattr(order, "client_order_id", "") or "")
        except (AttributeError, TypeError, ValueError) as exc:
            log.warning("trade_update.unparsed", event=event, error=str(exc))
            return
        if qty <= 0 or price <= 0.0 or side not in ("buy", "sell"):
            return
        await self.on_fill(symbol, side, qty, price, coid)

    async def run_forever(self) -> None:
        from alpaca.trading.stream import TradingStream

        backoff = 1.0
        while not self._stop.is_set():
            try:
                stream = TradingStream(self._api_key, self._secret_key,
                                       paper=self._paper)
                stream.subscribe_trade_updates(self._handle)
                log.info("trade_stream.connect")
                backoff = 1.0
                # alpaca-py's internal async runner
                await stream._run_forever()  # type: ignore[no-untyped-call]
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("trade_stream.disconnect", error=str(exc), retry_in=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def stop(self) -> None:
        self._stop.set()
