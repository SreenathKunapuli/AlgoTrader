"""Shared fixtures: in-memory repo, portfolio state, MockBroker."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from lobplatform.execution.broker import BrokerOrder
from lobplatform.persistence.repo import Repo
from lobplatform.risk.state import PortfolioState

# A known regular NYSE session: Monday 2026-06-15, 15:00 UTC = 11:00 ET.
IN_SESSION = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)


@pytest.fixture
def repo() -> Repo:
    return Repo("sqlite:///:memory:")


@pytest.fixture
def state() -> PortfolioState:
    s = PortfolioState()
    s.equity = 100_000.0
    s.cash = 100_000.0
    s.day_start_equity = 100_000.0
    s.peak_equity = 100_000.0
    return s


class MockBroker:
    """In-memory broker: fills market orders instantly at a settable price."""

    def __init__(self) -> None:
        self.price: dict[str, float] = {}
        self.positions: dict[str, int] = {}
        self.entry: dict[str, float] = {}
        self.open_orders: list[BrokerOrder] = []
        self.submitted: list[BrokerOrder] = []
        self.cancelled: list[str] = []
        self._cancelled_orders: dict[str, BrokerOrder] = {}
        self.closed_all = 0
        self._next = 0

    async def get_account(self):  # type: ignore[no-untyped-def]
        eq = 100_000.0 + sum(q * self.price.get(s, 0.0) for s, q in self.positions.items())
        return type("A", (), {"equity": eq, "cash": 100_000.0, "buying_power": eq})()

    async def get_positions(self):  # type: ignore[no-untyped-def]
        from lobplatform.execution.broker import BrokerPosition

        return [BrokerPosition(symbol=s, qty=q, avg_entry_price=self.entry.get(s, 0.0),
                               current_price=self.price.get(s, 0.0))
                for s, q in self.positions.items() if q != 0]

    async def get_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders)

    async def submit_order(self, symbol: str, side: str, qty: int, order_type: str,
                           client_order_id: str, limit_price: float | None = None) -> BrokerOrder:
        if any(o.client_order_id == client_order_id for o in self.submitted):
            raise RuntimeError("client_order_id must be unique")
        self._next += 1
        px = limit_price if limit_price is not None else self.price.get(symbol, 100.0)
        delta = qty if side == "buy" else -qty
        prev = self.positions.get(symbol, 0)
        self.positions[symbol] = prev + delta
        if prev == 0:
            self.entry[symbol] = px
        order = BrokerOrder(id=f"o{self._next}", client_order_id=client_order_id,
                            symbol=symbol, side=side, qty=qty, status="filled",
                            filled_qty=qty, filled_avg_price=px)
        self.submitted.append(order)
        return order

    async def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)
        for o in self.open_orders:
            if o.id == order_id:
                self._cancelled_orders[order_id] = BrokerOrder(
                    id=o.id, client_order_id=o.client_order_id, symbol=o.symbol,
                    side=o.side, qty=o.qty, status="canceled",
                    filled_qty=o.filled_qty, filled_avg_price=o.filled_avg_price)
        self.open_orders = [o for o in self.open_orders if o.id != order_id]

    async def get_order(self, order_id: str) -> BrokerOrder:
        if order_id in self._cancelled_orders:
            return self._cancelled_orders[order_id]
        for o in self.submitted + self.open_orders:
            if o.id == order_id:
                return o
        raise KeyError(order_id)

    async def close_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)

    async def close_all_positions(self) -> None:
        self.closed_all += 1
        self.positions.clear()
        self.open_orders.clear()


@pytest.fixture
def mock_broker() -> MockBroker:
    return MockBroker()
