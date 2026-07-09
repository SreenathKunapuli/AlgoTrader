"""In-memory portfolio state shared by risk/execution, synced from broker.

Why: the risk manager needs equity, positions, and day/peak anchors on
every approval without a broker round-trip; reconcile.py and fill events
keep this mirror honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Position:
    symbol: str
    qty: int                 # signed: + long, − short
    entry_price: float
    mark: float
    stop_price: float | None = None
    entry_ts: datetime | None = None
    entry_signals: dict[str, float] = field(default_factory=dict)
    book: str = "intraday"   # "intraday" (ensemble) | "xsec" (monthly momentum)

    @property
    def market_value(self) -> float:
        return self.qty * self.mark

    @property
    def unrealized_pnl(self) -> float:
        return (self.mark - self.entry_price) * self.qty


@dataclass
class PortfolioState:
    equity: float = 0.0
    cash: float = 0.0
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    last_data_ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    halted: bool = False
    halted_reason: str = ""

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    def book_gross(self, book: str) -> float:
        return sum(abs(p.market_value) for p in self.positions.values()
                   if p.book == book)

    def book_positions(self, book: str) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.book == book}

    @property
    def day_pnl(self) -> float:
        return self.equity - self.day_start_equity if self.day_start_equity else 0.0

    @property
    def day_pnl_pct(self) -> float:
        return self.day_pnl / self.day_start_equity if self.day_start_equity else 0.0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
