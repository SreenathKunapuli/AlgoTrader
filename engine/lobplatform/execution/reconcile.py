"""Startup reconciliation: broker is the source of truth.

Fetch broker positions + open orders, diff against our DB/state mirror,
adopt broker values, and log every discrepancy. Idempotent order IDs plus
this reconcile step make restarts safe mid-session.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from ..persistence.repo import Repo
from ..risk.state import PortfolioState, Position
from .broker import AlpacaBroker

log = structlog.get_logger()


async def reconcile(broker: AlpacaBroker, repo: Repo, state: PortfolioState) -> None:
    account = await broker.get_account()
    state.equity = account.equity
    state.cash = account.cash

    broker_positions = {p.symbol: p for p in await broker.get_positions()}
    ours = set(state.positions)
    theirs = set(broker_positions)

    for sym in theirs - ours:
        p = broker_positions[sym]
        log.warning("reconcile.adopt_position", symbol=sym, qty=p.qty)
        state.positions[sym] = Position(symbol=sym, qty=p.qty,
                                        entry_price=p.avg_entry_price,
                                        mark=p.current_price or p.avg_entry_price)
    for sym in ours - theirs:
        log.warning("reconcile.drop_stale_position", symbol=sym)
        del state.positions[sym]
    for sym in ours & theirs:
        p, q = broker_positions[sym], state.positions[sym]
        if p.qty != q.qty:
            log.warning("reconcile.qty_mismatch", symbol=sym, ours=q.qty, broker=p.qty)
            q.qty = p.qty
            q.entry_price = p.avg_entry_price
        q.mark = p.current_price or q.mark

    open_orders = await broker.get_open_orders()
    for o in open_orders:
        repo.upsert_order(o.client_order_id or o.id, broker_order_id=o.id,
                          symbol=o.symbol, side=o.side, qty=o.qty,
                          order_type="unknown", status=o.status,
                          ts=datetime.now(UTC), filled_qty=o.filled_qty)
    log.info("reconcile.done", positions=len(state.positions), open_orders=len(open_orders))
