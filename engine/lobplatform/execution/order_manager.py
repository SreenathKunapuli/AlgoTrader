"""OrderManager — the sole route to the broker; requires Approval tokens.

Order policy (§ decision table): limit at mid ± 10% of spread (buy above
mid, sell below — toward the touch); unfilled after 30s -> cancel-replace
as market. TIF=DAY. Idempotent client_order_id =
sha1(f"{strategy}:{symbol}:{side}:{bar_ts_iso}")[:32].

emergency_* methods are the kill switch's path — they go straight to the
broker's cancel/close endpoints and bypass entry logic (never entry risk).
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime

import structlog

from ..persistence.repo import Repo
from ..risk.risk_manager import Approval
from ..risk.state import PortfolioState, Position
from .broker import AlpacaBroker, BrokerOrder

log = structlog.get_logger()

UNFILLED_REPLACE_S = 30.0


def make_client_order_id(strategy: str, symbol: str, side: str, bar_ts: datetime) -> str:
    raw = f"{strategy}:{symbol}:{side}:{bar_ts.isoformat()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:32]


class OrderManager:
    def __init__(self, broker: AlpacaBroker, repo: Repo, state: PortfolioState) -> None:
        self._broker = broker
        self.repo = repo
        self.state = state

    async def submit(self, approval: Approval, bar_ts: datetime,
                     mid: float, spread: float, strategy: str = "ensemble") -> BrokerOrder | None:
        """Submit an approved intent as a limit order with market fallback."""
        intent = approval.intent
        coid = make_client_order_id(strategy, intent.symbol, intent.side, bar_ts)
        offset = 0.10 * max(spread, 0.0)
        limit_price = mid + offset if intent.side == "buy" else mid - offset
        try:
            order = await self._broker.submit_order(
                symbol=intent.symbol, side=intent.side, qty=intent.qty,
                order_type="limit", client_order_id=coid,
                limit_price=limit_price,
            )
        except Exception as exc:
            if "client_order_id must be unique" in str(exc).lower():
                log.info("order.duplicate_suppressed", coid=coid)
                return None
            raise
        self.repo.upsert_order(coid, broker_order_id=order.id, symbol=intent.symbol,
                               side=intent.side, qty=intent.qty, order_type="limit",
                               limit_price=limit_price, status=order.status,
                               ts=datetime.now(UTC), reason=intent.reason)
        asyncio.get_event_loop().call_later(
            UNFILLED_REPLACE_S,
            lambda: asyncio.ensure_future(self._replace_if_unfilled(order, intent.reason)),
        )
        return order

    async def _replace_if_unfilled(self, order: BrokerOrder, reason: str) -> None:
        """Cancel-replace as market if the limit hasn't fully filled in 30s."""
        try:
            open_orders = await self._broker.get_open_orders()
            live = next((o for o in open_orders if o.id == order.id), None)
            if live is None:
                return  # filled or already cancelled
            await self._broker.cancel_order(order.id)
            remaining = live.qty - live.filled_qty
            if remaining <= 0:
                return
            coid = f"{order.client_order_id[:24]}-mkt"
            mkt = await self._broker.submit_order(
                symbol=order.symbol, side=order.side, qty=remaining,
                order_type="market", client_order_id=coid)
            self.repo.upsert_order(coid, broker_order_id=mkt.id, symbol=order.symbol,
                                   side=order.side, qty=remaining, order_type="market",
                                   status=mkt.status, ts=datetime.now(UTC),
                                   reason=reason)
            log.info("order.cancel_replace_market", symbol=order.symbol, qty=remaining)
        except Exception as exc:
            log.error("order.replace_failed", error=str(exc))

    # --- fill bookkeeping (called from trade-updates stream / polling) --- #
    def on_fill(self, symbol: str, side: str, qty: int, price: float,
                reason: str, signals: dict[str, float] | None = None) -> None:
        """Update the position mirror and record round-trips on close."""
        delta = qty if side == "buy" else -qty
        pos = self.state.positions.get(symbol)
        now = datetime.now(UTC)
        if pos is None or pos.qty == 0:
            self.state.positions[symbol] = Position(
                symbol=symbol, qty=delta, entry_price=price, mark=price,
                entry_ts=now, entry_signals=signals or {})
            return
        new_qty = pos.qty + delta
        closing = (pos.qty > 0 > delta) or (pos.qty < 0 < delta)
        if closing:
            closed = min(abs(delta), abs(pos.qty))
            side_str = "long" if pos.qty > 0 else "short"
            pnl = (price - pos.entry_price) * closed * (1 if pos.qty > 0 else -1)
            self.repo.add_trade(symbol=symbol, side=side_str, qty=closed,
                                entry_ts=pos.entry_ts or now, exit_ts=now,
                                entry_price=pos.entry_price, exit_price=price,
                                pnl=pnl, signal_scores_json=pos.entry_signals)
        if new_qty == 0:
            del self.state.positions[symbol]
        else:
            pos.qty = new_qty
            pos.mark = price
            if not closing:  # scale-in: blend entry
                pos.entry_price = (pos.entry_price * (new_qty - delta) + price * delta) / new_qty

    # --- kill-switch emergency path --- #
    async def emergency_cancel_all(self) -> None:
        try:
            for o in await self._broker.get_open_orders():
                await self._broker.cancel_order(o.id)
        except Exception as exc:
            log.error("emergency_cancel_failed", error=str(exc))

    async def emergency_flatten_all(self) -> None:
        try:
            await self._broker.close_all_positions()
        except Exception as exc:
            log.error("emergency_flatten_failed", error=str(exc))

    async def verify_flat(self) -> bool:
        try:
            positions = await self._broker.get_positions()
            if not positions:
                self.state.positions.clear()
            return not positions
        except Exception:
            return False
