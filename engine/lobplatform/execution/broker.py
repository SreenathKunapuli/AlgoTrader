"""Alpaca broker wrapper — the ONLY module that touches the trading API.

Paper-mode enforcement lives here, at construction time: the base URL must
contain "paper-api" or the process refuses to start. (Phase 4 live mode
adds separate gated factories; until then this is absolute.)

Exposes exactly the seven calls the platform needs. A token bucket keeps
us under 150 req/min as a courtesy to the free tier.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger()

PAPER_URL = "https://paper-api.alpaca.markets"


class PaperModeViolation(RuntimeError):
    pass


def assert_paper(base_url: str) -> None:
    """Hard startup assertion: refuse anything that is not the paper host."""
    if "paper-api" not in base_url:
        raise PaperModeViolation(
            f"TRADING SAFETY: base_url={base_url!r} is not the Alpaca paper endpoint. "
            "This build only trades paper. Refusing to start."
        )


@dataclass
class BrokerAccount:
    equity: float
    cash: float
    buying_power: float


@dataclass
class BrokerPosition:
    symbol: str
    qty: int              # signed
    avg_entry_price: float
    current_price: float


@dataclass
class BrokerOrder:
    id: str
    client_order_id: str
    symbol: str
    side: str
    qty: int
    status: str
    filled_qty: int = 0
    filled_avg_price: float | None = None


class _TokenBucket:
    def __init__(self, rate_per_min: int = 150) -> None:
        self._interval = 60.0 / rate_per_min
        self._next_ok = 0.0

    async def acquire(self) -> None:
        now = time.monotonic()
        if now < self._next_ok:
            await asyncio.sleep(self._next_ok - now)
        self._next_ok = max(now, self._next_ok) + self._interval


class AlpacaBroker:
    """Thin async facade over alpaca-py's TradingClient (which is sync)."""

    def __init__(self, api_key: str, secret_key: str, base_url: str = PAPER_URL) -> None:
        assert_paper(base_url)
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(api_key, secret_key, paper=True)
        self._bucket = _TokenBucket()

    async def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        await self._bucket.acquire()
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def get_account(self) -> BrokerAccount:
        a = await self._call(self._client.get_account)
        return BrokerAccount(equity=float(a.equity), cash=float(a.cash),
                             buying_power=float(a.buying_power))

    async def get_positions(self) -> list[BrokerPosition]:
        rows = await self._call(self._client.get_all_positions)
        return [BrokerPosition(symbol=p.symbol,
                               qty=int(float(p.qty)) * (1 if p.side.value == "long" else -1),
                               avg_entry_price=float(p.avg_entry_price),
                               current_price=float(p.current_price or 0.0)) for p in rows]

    async def get_open_orders(self) -> list[BrokerOrder]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        rows = await self._call(self._client.get_orders,
                                GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return [self._map_order(o) for o in rows]

    async def submit_order(self, symbol: str, side: str, qty: int, order_type: str,
                           client_order_id: str, limit_price: float | None = None) -> BrokerOrder:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side_e = OrderSide.BUY if side == "buy" else OrderSide.SELL
        req: Any
        if order_type == "limit" and limit_price is not None:
            req = LimitOrderRequest(symbol=symbol, qty=qty, side=side_e,
                                    time_in_force=TimeInForce.DAY,
                                    limit_price=round(limit_price, 2),
                                    client_order_id=client_order_id)
        else:
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=side_e,
                                     time_in_force=TimeInForce.DAY,
                                     client_order_id=client_order_id)
        o = await self._call(self._client.submit_order, req)
        return self._map_order(o)

    async def cancel_order(self, order_id: str) -> None:
        await self._call(self._client.cancel_order_by_id, order_id)

    async def get_order(self, order_id: str) -> BrokerOrder:
        o = await self._call(self._client.get_order_by_id, order_id)
        return self._map_order(o)

    async def close_position(self, symbol: str) -> None:
        await self._call(self._client.close_position, symbol)

    async def close_all_positions(self) -> None:
        await self._call(self._client.close_all_positions, cancel_orders=True)

    @staticmethod
    def _map_order(o: Any) -> BrokerOrder:
        return BrokerOrder(
            id=str(o.id), client_order_id=str(o.client_order_id or ""),
            symbol=o.symbol, side=o.side.value, qty=int(float(o.qty or 0)),
            status=o.status.value, filled_qty=int(float(o.filled_qty or 0)),
            filled_avg_price=float(o.filled_avg_price) if o.filled_avg_price else None,
        )
