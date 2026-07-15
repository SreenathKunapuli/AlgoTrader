"""Kill switch — built before the engine, wired into everything.

Two scopes since 2026-07-10 (see config/tiers.py for the why):

GLOBAL (fire): account drawdown floor; 5 broker errors in 60s; manual.
Action sequence: persist HALTED -> cancel all orders -> flatten (market)
-> verify flat with up to 3 retries -> emit CRITICAL event. HALTED
survives restarts (engine_state row); only `lobctl reset` clears it.

INTRADAY (fire_intraday): intraday-book daily loss; data staleness >180s
in-session. Flattens and halts the INTRADAY book only — the monthly xsec
book neither needs the live stream nor deserves liquidation for an
intraday breach. Cleared automatically at the next day roll.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import structlog

from ..config.tiers import TierConfig
from ..data import calendar
from ..persistence.repo import Repo
from .state import PortfolioState

log = structlog.get_logger()

EmitFn = Callable[[str, dict[str, object]], Awaitable[None]]


class KillSwitch:
    def __init__(self, state: PortfolioState, tier: TierConfig, repo: Repo,
                 order_manager: object, emit: EmitFn,
                 staleness_kill_s: int = 180,
                 broker_error_count: int = 5, broker_error_window_s: int = 60,
                 flatten_intraday: Callable[[str], Awaitable[None]] | None = None,
                 ) -> None:
        # order_manager typed loosely to avoid an import cycle; it must expose
        # emergency_cancel_all() / emergency_flatten_all() / verify_flat().
        # flatten_intraday is the engine's graceful intraday-book exit path
        # (urgent limit -> market), injected to avoid the same cycle.
        self.state = state
        self.tier = tier
        self.repo = repo
        self.om = order_manager
        self.emit = emit
        self.staleness_kill_s = staleness_kill_s
        self.flatten_intraday = flatten_intraday
        self._error_times: deque[float] = deque(maxlen=broker_error_count)
        self._error_threshold = broker_error_count
        self._error_window = broker_error_window_s

    def record_broker_error(self) -> bool:
        """Track an error; True if the error-rate trigger fired."""
        now = time.monotonic()
        self._error_times.append(now)
        if (len(self._error_times) == self._error_threshold
                and now - self._error_times[0] <= self._error_window):
            return True
        return False

    def check_triggers(self, now: datetime | None = None
                       ) -> tuple[str, str] | None:
        """Return (scope, reason) — scope "account" or "intraday" — or None.
        Called every bar/monitor tick."""
        now = now or datetime.now(UTC)
        s, t = self.state, self.tier
        if s.drawdown_pct >= t.max_drawdown_pct:
            return ("account", f"account drawdown floor: {s.drawdown_pct:.2%}"
                               f" >= {t.max_drawdown_pct:.2%}")
        if s.day_start_equity > 0 and s.intraday_day_pnl_pct <= -t.daily_loss_limit_pct:
            return ("intraday", f"intraday daily loss: {s.intraday_day_pnl_pct:.2%}"
                                f" <= -{t.daily_loss_limit_pct:.2%}")
        stale = (now - s.last_data_ts).total_seconds()
        if calendar.is_session_open(now) and stale > self.staleness_kill_s:
            return ("intraday", f"data staleness: {stale:.0f}s > {self.staleness_kill_s}s")
        return None

    async def fire(self, reason: str) -> None:
        """Execute the full kill sequence. Idempotent."""
        if self.state.halted:
            return
        # (1) persist HALTED first — even if flattening fails, we stay halted
        self.state.halted = True
        self.state.halted_reason = reason
        self.repo.update_state(status="HALTED", halted_reason=reason)
        log.critical("kill_switch.fired", reason=reason)
        # (2) cancel all open orders
        await self.om.emergency_cancel_all()          # type: ignore[attr-defined]
        # (3) flatten all positions with market orders
        await self.om.emergency_flatten_all()         # type: ignore[attr-defined]
        # (4) verify flat, retry up to 3x
        for attempt in range(3):
            if await self.om.verify_flat():           # type: ignore[attr-defined]
                break
            log.error("kill_switch.not_flat_retry", attempt=attempt + 1)
            await self.om.emergency_flatten_all()     # type: ignore[attr-defined]
        # (5) emit event
        await self.emit("engine_status",
                        {"status": "HALTED", "reason": reason,
                         "ts": datetime.now(UTC).isoformat()})

    async def fire_intraday(self, reason: str) -> None:
        """Intraday-scoped kill: flatten and halt the intraday book only.
        Day-scoped (day_roll clears it); never persists as HALTED."""
        if self.state.halted or self.state.intraday_halted:
            return
        self.state.intraday_halted = True
        log.critical("kill_switch.intraday_fired", reason=reason)
        self.repo.update_state(status="INTRADAY_HALTED", halted_reason=reason)
        if self.flatten_intraday is not None:
            await self.flatten_intraday(reason)
        await self.emit("engine_status",
                        {"status": "INTRADAY_HALTED", "reason": reason,
                         "ts": datetime.now(UTC).isoformat()})
