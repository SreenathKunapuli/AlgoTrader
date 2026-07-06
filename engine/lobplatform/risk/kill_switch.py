"""Kill switch — built before the engine, wired into everything.

Triggers (§5.4): daily loss limit; max drawdown; data staleness >180s
in-session; 5 broker errors in 60s; manual. Action sequence: persist
HALTED -> cancel all orders -> flatten (market) -> verify flat with up to
3 retries -> emit CRITICAL event. HALTED survives restarts (engine_state
row); only `lobctl reset` / POST /engine/reset clears it.
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
                 broker_error_count: int = 5, broker_error_window_s: int = 60) -> None:
        # order_manager typed loosely to avoid an import cycle; it must expose
        # emergency_cancel_all() / emergency_flatten_all() / verify_flat().
        self.state = state
        self.tier = tier
        self.repo = repo
        self.om = order_manager
        self.emit = emit
        self.staleness_kill_s = staleness_kill_s
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

    def check_triggers(self, now: datetime | None = None) -> str | None:
        """Return a trigger reason or None. Called every bar/monitor tick."""
        now = now or datetime.now(UTC)
        s, t = self.state, self.tier
        if s.day_start_equity > 0 and s.day_pnl_pct <= -t.daily_loss_limit_pct:
            return f"daily loss limit: {s.day_pnl_pct:.2%} <= -{t.daily_loss_limit_pct:.2%}"
        if s.drawdown_pct >= t.max_drawdown_pct:
            return f"max drawdown: {s.drawdown_pct:.2%} >= {t.max_drawdown_pct:.2%}"
        stale = (now - s.last_data_ts).total_seconds()
        if calendar.is_session_open(now) and stale > self.staleness_kill_s:
            return f"data staleness: {stale:.0f}s > {self.staleness_kill_s}s"
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
