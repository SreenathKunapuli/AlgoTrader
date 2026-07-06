"""Live shadow evaluation (§5.9): per-signal realized performance tracking.

Attribution model: a closed trade's P&L is attributed to each signal in
proportion to its (weight x score x confidence) contribution at entry
(stored on the trade row). Rolling 20-session hit rate < 50% or negative
attributed P&L over 20 sessions -> halve that signal's ensemble weight,
persist a flagged signal_health row, and emit a warning. Weight restores
only when a retrained model passes the quality gate (train path resets
multipliers).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select

from ..persistence.models import Trade
from ..persistence.repo import Repo

log = structlog.get_logger()

ROLLING_SESSIONS = 20
HIT_RATE_FLOOR = 0.50


class SignalHealthTracker:
    def __init__(self, repo: Repo, signal_names: list[str]) -> None:
        self.repo = repo
        self.signal_names = signal_names
        self.multipliers: dict[str, float] = dict.fromkeys(signal_names, 1.0)

    def evaluate(self, now: datetime | None = None) -> dict[str, float]:
        """Recompute rolling stats; update and persist multipliers."""
        now = now or datetime.now(UTC)
        since = now - timedelta(days=ROLLING_SESSIONS * 1.5)  # calendar padding
        with self.repo.session() as s:
            trades = list(s.scalars(select(Trade).where(Trade.exit_ts >= since)))
        for name in self.signal_names:
            attributed: list[float] = []
            for t in trades:
                scores = t.signal_scores_json or {}
                contribs = {
                    k: abs(float(v.get("contribution", 0.0)))
                    if isinstance(v, dict) else abs(float(v))
                    for k, v in scores.items()
                }
                total = sum(contribs.values())
                if total <= 0 or name not in contribs:
                    continue
                attributed.append(t.pnl * contribs[name] / total)
            if len(attributed) < 5:      # not enough evidence to judge
                continue
            hit = sum(1 for p in attributed if p > 0) / len(attributed)
            pnl = sum(attributed)
            degraded = hit < HIT_RATE_FLOOR or pnl < 0
            new_mult = 0.5 if degraded else self.multipliers.get(name, 1.0)
            if degraded and self.multipliers.get(name, 1.0) > new_mult:
                log.warning("signal_health.degraded", signal=name,
                            hit_rate=round(hit, 3), pnl=round(pnl, 2))
            self.multipliers[name] = new_mult
            self.repo.add_signal_health(
                signal=name, rolling_hit_rate=hit, attributed_pnl_20s=pnl,
                weight_multiplier=new_mult, flagged_for_retrain=degraded)
        return self.multipliers

    def restore(self, name: str) -> None:
        """Called after a retrained model passes the quality gate."""
        self.multipliers[name] = 1.0
