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
    xsec_qty: int = 0        # shares owned by the xsec book; intraday addon = qty - xsec_qty

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
    # scoped halts (2026-07-10): a global kill was the only lever, so an
    # intraday-book breach liquidated the monthly book too. These are
    # day/cycle-scoped and never persist as HALTED.
    intraday_halted: bool = False        # intraday book: no entries, flattened
    xsec_buys_halted: bool = False       # xsec circuit breaker: sells only
    intraday_realized_today: float = 0.0  # closed intraday PnL since day roll

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    def book_gross(self, book: str) -> float:
        return sum(abs(p.market_value) for p in self.positions.values()
                   if p.book == book)

    def book_positions(self, book: str) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.book == book}

    def book_unrealized(self, book: str) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values()
                   if p.book == book)

    def book_cost_basis(self, book: str) -> float:
        return sum(abs(p.qty * p.entry_price) for p in self.positions.values()
                   if p.book == book)

    @property
    def intraday_day_pnl_pct(self) -> float:
        """Intraday-book day PnL: realized today + open unrealized. (For the
        LOW tier, which holds overnight, unrealized-since-entry overstates
        the day component — a conservative approximation.)"""
        if not self.day_start_equity:
            return 0.0
        pnl = self.intraday_realized_today + self.book_unrealized("intraday")
        return pnl / self.day_start_equity

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
