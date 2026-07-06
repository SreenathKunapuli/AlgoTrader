"""RiskManager — every order passes through approve(); no exceptions.

Structural enforcement: OrderManager.submit() requires an `Approval` token,
and Approval's constructor is module-private by convention (`_issue`), so
the only way to obtain one is this class. Exits (stop / EOD / kill-flatten)
use approve_exit(), which skips *entry-only* checks (windows, position
caps) but still refuses in nonsensical states.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from ..config.tiers import TierConfig
from ..data import calendar
from .state import PortfolioState

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    qty: int
    price_hint: float                 # latest mark, for exposure math
    reason: str = "signal"            # signal / stop / eod / kill

    def as_dict(self) -> dict[str, object]:
        return {"symbol": self.symbol, "side": self.side, "qty": self.qty,
                "price_hint": self.price_hint, "reason": self.reason}


@dataclass(frozen=True)
class Approval:
    intent: OrderIntent
    token: str
    issued_at: datetime


@dataclass(frozen=True)
class Rejection:
    intent: OrderIntent
    reason: str


class RiskManager:
    def __init__(self, tier: TierConfig, state: PortfolioState) -> None:
        self.tier = tier
        self.state = state

    def _issue(self, intent: OrderIntent) -> Approval:
        return Approval(intent=intent, token=secrets.token_hex(8),
                        issued_at=datetime.now(UTC))

    def _post_trade_position_value(self, intent: OrderIntent) -> float:
        pos = self.state.positions.get(intent.symbol)
        cur_qty = pos.qty if pos else 0
        delta = intent.qty if intent.side == "buy" else -intent.qty
        return abs((cur_qty + delta) * intent.price_hint)

    def _post_trade_gross(self, intent: OrderIntent) -> float:
        pos = self.state.positions.get(intent.symbol)
        cur_val = abs(pos.market_value) if pos else 0.0
        return self.state.gross_exposure - cur_val + self._post_trade_position_value(intent)

    def _is_entry(self, intent: OrderIntent) -> bool:
        """Entry = increases absolute exposure in the symbol."""
        pos = self.state.positions.get(intent.symbol)
        cur_qty = pos.qty if pos else 0
        delta = intent.qty if intent.side == "buy" else -intent.qty
        return abs(cur_qty + delta) > abs(cur_qty)

    def approve(self, intent: OrderIntent,
                now: datetime | None = None) -> Approval | Rejection:
        """Full check sequence per §5.4, in order. Entry path."""
        now = now or datetime.now(UTC)
        s, t = self.state, self.tier
        if s.halted:
            return Rejection(intent, "engine HALTED")
        if intent.symbol not in t.universe:
            return Rejection(intent, f"symbol {intent.symbol} not in tier universe")
        if intent.qty <= 0:
            return Rejection(intent, "non-positive quantity")
        is_entry = self._is_entry(intent)
        if is_entry:
            pos = s.positions.get(intent.symbol)
            cur_qty = pos.qty if pos else 0
            delta = intent.qty if intent.side == "buy" else -intent.qty
            if (cur_qty + delta) < 0 and not t.allow_short:
                return Rejection(intent, "shorting not allowed in this tier")
            if self._post_trade_position_value(intent) > t.max_position_pct * s.equity + 1e-6:
                return Rejection(intent, "exceeds max position size")
            if self._post_trade_gross(intent) > t.max_gross_pct * s.equity + 1e-6:
                return Rejection(intent, "exceeds max gross exposure")
            new_symbol = intent.symbol not in s.positions
            if new_symbol and len(s.positions) >= t.max_open_positions:
                return Rejection(intent, "exceeds max open positions")
            if s.day_pnl_pct <= -t.daily_loss_limit_pct:
                return Rejection(intent, "daily loss limit reached")
            if s.drawdown_pct >= t.max_drawdown_pct:
                return Rejection(intent, "max drawdown reached")
            if not calendar.in_entry_window(now):
                return Rejection(intent, "outside entry window")
        return self._issue(intent)

    def approve_exit(self, intent: OrderIntent) -> Approval | Rejection:
        """Stops / EOD / kill flatten: risk-checked but window/cap-exempt."""
        if intent.qty <= 0:
            return Rejection(intent, "non-positive quantity")
        if not self._is_entry(intent):
            return self._issue(intent)
        return Rejection(intent, "exit path used for an exposure-increasing order")
