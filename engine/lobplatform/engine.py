"""The engine: asyncio loop wiring data -> signals -> risk -> execution.

Task graph (§5.6):
  stream consumer -> BarBuilder(1m) -> aggregate(5m) -> on tier cadence:
  signals -> ensemble -> target portfolio -> diff -> intents -> RiskManager
  -> OrderManager. Independent tasks: stop monitor, staleness monitor, EOD
  flattener, heartbeat persister, command poller (API control channel).

Testability: the engine takes its broker/stream via constructor injection;
the replay test drives `on_minute_bar` directly with a MockBroker.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import structlog

from .config.settings import Settings
from .config.tiers import TIERS, Tier, TierConfig
from .data import calendar
from .data.bar_builder import Bar, BarBuilder, aggregate
from .execution.order_manager import OrderManager
from .persistence.repo import Repo
from .pubsub import PubSub
from .risk.kill_switch import KillSwitch
from .risk.risk_manager import OrderIntent, Rejection, RiskManager
from .risk.sizing import size_position
from .risk.state import PortfolioState
from .signals.ensemble import Ensemble
from .signals.health import SignalHealthTracker

log = structlog.get_logger()

ATR_PERIOD = 14
MAX_5M_BARS = 2400  # ~1 month of 5-min bars kept in memory per symbol


def atr_from_bars(bars: list[Bar], period: int = ATR_PERIOD) -> float:
    if len(bars) < 2:
        return 0.0
    highs = np.array([b.high for b in bars[-(period + 1):]])
    lows = np.array([b.low for b in bars[-(period + 1):]])
    closes = np.array([b.close for b in bars[-(period + 1):]])
    prev = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev), np.abs(lows - prev)))
    return float(tr.mean())


class Engine:
    def __init__(self, settings: Settings, tier: TierConfig, repo: Repo,
                 order_manager: OrderManager, ensemble: Ensemble,
                 state: PortfolioState, pubsub: PubSub) -> None:
        self.settings = settings
        self.tier = tier
        self.repo = repo
        self.om = order_manager
        self.ensemble = ensemble
        self.state = state
        self.pubsub = pubsub
        self.risk = RiskManager(tier, state)
        self.kill = KillSwitch(state, tier, repo, order_manager, pubsub_emit(pubsub),
                               staleness_kill_s=settings.staleness_kill_s,
                               broker_error_count=settings.broker_error_kill_count,
                               broker_error_window_s=settings.broker_error_kill_window_s)
        self.builder = BarBuilder(interval_s=settings.bar_interval_s)
        self.bars_1m: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=12000))
        self.bars_5m: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=MAX_5M_BARS))
        self._pending_1m: dict[str, list[Bar]] = defaultdict(list)
        self._last_rebalance = datetime.min.replace(tzinfo=UTC)
        self.health = SignalHealthTracker(repo, list(ensemble.signals))
        self._paused_stale = False

    # ---------------- data path ---------------- #
    def warmup(self, history: dict[str, list[Bar]]) -> None:
        """Seed bar caches from REST backfill (1-min bars)."""
        for sym, bars in history.items():
            self.bars_1m[sym].extend(bars)
            group: list[Bar] = []
            for b in bars:
                group.append(b)
                if len(group) == 5:
                    agg = aggregate(group, 300)
                    if agg:
                        self.bars_5m[sym].append(agg)
                    group = []
        log.info("warmup.done", symbols=len(history))

    async def on_trade(self, symbol: str, ts: datetime, price: float, size: int) -> None:
        self.state.last_data_ts = datetime.now(UTC)
        done = self.builder.on_trade(symbol, ts, price, size)
        if done:
            await self.on_minute_bar(done)
        pos = self.state.positions.get(symbol)
        if pos:
            pos.mark = price

    async def on_quote(self, symbol: str, ts: datetime, bid: float, bid_sz: int,
                       ask: float, ask_sz: int) -> None:
        self.state.last_data_ts = datetime.now(UTC)
        done = self.builder.on_quote(symbol, ts, bid, bid_sz, ask, ask_sz)
        if done:
            await self.on_minute_bar(done)

    async def on_minute_bar(self, bar: Bar) -> None:
        """Finalized 1-min bar: cache, aggregate to 5-min, maybe act."""
        self.bars_1m[bar.symbol].append(bar)
        pend = self._pending_1m[bar.symbol]
        pend.append(bar)
        if len(pend) >= 5:
            five = aggregate(pend[:5], 300)
            del pend[:5]
            if five:
                self.bars_5m[bar.symbol].append(five)
                await self.on_five_min_bar(five)

    async def on_five_min_bar(self, bar: Bar) -> None:
        await self.check_stops(bar)
        reason = self.kill.check_triggers()
        if reason and not self.state.halted:
            await self.kill.fire(reason)
            return
        now = datetime.now(UTC)
        if (now - self._last_rebalance).total_seconds() >= self.tier.rebalance_seconds:
            self._last_rebalance = now
            await self.rebalance(now)

    # ---------------- decision path ---------------- #
    async def rebalance(self, now: datetime) -> None:
        if self.state.halted or self._paused_stale:
            return
        if not calendar.in_entry_window(now):
            return
        for symbol in self.tier.universe:
            bars = list(self.bars_5m.get(symbol, ()))
            if len(bars) < 30:
                continue
            res = self.ensemble.compute(symbol, bars, self.tier)
            self.repo.add_signal(now, symbol, res.per_signal, res.final_score)
            await self.pubsub.publish("signals", {
                "symbol": symbol, "ensemble": res.final_score,
                "per_signal": res.per_signal, "ts": now.isoformat()})
            if not res.is_candidate(self.tier):
                # candidate exit: existing position whose signal died
                await self._maybe_exit_on_signal(symbol, bars, res.final_score)
                continue
            await self._enter_or_adjust(symbol, bars, res, now)

    async def _maybe_exit_on_signal(self, symbol: str, bars: list[Bar],
                                    score: float) -> None:
        pos = self.state.positions.get(symbol)
        if not pos:
            return
        # exit when the signal flips against the position or dies
        if (pos.qty > 0 and score <= 0) or (pos.qty < 0 and score >= 0):
            await self._exit_position(symbol, "signal")

    async def _enter_or_adjust(self, symbol: str, bars: list[Bar],
                               res: Any, now: datetime) -> None:
        price = bars[-1].close
        atr = atr_from_bars(bars)
        target_qty = size_position(self.tier, self.state.equity, price, atr, res.vol_mult)
        if res.final_score < 0:
            target_qty = -target_qty if self.tier.allow_short else 0
        pos = self.state.positions.get(symbol)
        cur_qty = pos.qty if pos else 0
        delta = target_qty - cur_qty
        if delta == 0 or target_qty == 0 and cur_qty == 0:
            return
        side: Literal["buy", "sell"] = "buy" if delta > 0 else "sell"
        intent = OrderIntent(symbol=symbol, side=side, qty=abs(delta),
                             price_hint=price, reason="signal")
        approval = self.risk.approve(intent, now)
        if isinstance(approval, Rejection):
            self.repo.add_rejection(approval.reason, intent.as_dict())
            log.info("risk.rejected", symbol=symbol, reason=approval.reason)
            return
        spread = bars[-1].mean_spread
        try:
            order = await self.om.submit(approval, bars[-1].ts, mid=price, spread=spread)
        except Exception as exc:
            log.error("order.submit_failed", symbol=symbol, error=str(exc))
            if self.kill.record_broker_error():
                await self.kill.fire("5 consecutive broker errors in 60s")
            return
        if order:
            stop_mult = self.tier.stop_atr_multiple
            stop = price - stop_mult * atr if delta > 0 else price + stop_mult * atr
            # record intended stop; on_fill will attach it to the position
            p = self.state.positions.get(symbol)
            if p:
                p.stop_price = stop
            await self.pubsub.publish("orders", {
                "symbol": symbol, "side": side, "qty": abs(delta),
                "ts": now.isoformat(), "reason": "signal"})

    async def _exit_position(self, symbol: str, reason: str) -> None:
        pos = self.state.positions.get(symbol)
        if not pos or pos.qty == 0:
            return
        side: Literal["buy", "sell"] = "sell" if pos.qty > 0 else "buy"
        intent = OrderIntent(symbol=symbol, side=side, qty=abs(pos.qty),
                             price_hint=pos.mark, reason=reason)
        approval = self.risk.approve_exit(intent)
        if isinstance(approval, Rejection):
            self.repo.add_rejection(approval.reason, intent.as_dict())
            return
        try:
            await self.om.submit(approval, datetime.now(UTC),
                                 mid=pos.mark, spread=0.0)
        except Exception as exc:
            log.error("exit.submit_failed", symbol=symbol, error=str(exc))

    async def check_stops(self, bar: Bar) -> None:
        pos = self.state.positions.get(bar.symbol)
        if not pos or pos.stop_price is None:
            return
        hit = (pos.qty > 0 and bar.low <= pos.stop_price) or \
              (pos.qty < 0 and bar.high >= pos.stop_price)
        if hit:
            log.info("stop.hit", symbol=bar.symbol, stop=pos.stop_price)
            await self._exit_position(bar.symbol, "stop")

    # ---------------- background tasks ---------------- #
    async def staleness_monitor(self) -> None:
        while True:
            await asyncio.sleep(5)
            now = datetime.now(UTC)
            if not calendar.is_session_open(now) or self.state.halted:
                continue
            stale = (now - self.state.last_data_ts).total_seconds()
            if stale > self.settings.staleness_kill_s:
                await self.kill.fire(f"data staleness {stale:.0f}s")
            elif stale > self.settings.staleness_pause_s and not self._paused_stale:
                self._paused_stale = True
                self.repo.update_state(status="PAUSED")
                await self.pubsub.publish("engine_status",
                                          {"status": "PAUSED", "reason": "stale data"})
            elif stale <= self.settings.staleness_pause_s and self._paused_stale:
                self._paused_stale = False
                self.repo.update_state(status="RUNNING")
                await self.pubsub.publish("engine_status", {"status": "RUNNING"})

    async def eod_flattener(self) -> None:
        while True:
            await asyncio.sleep(20)
            now = datetime.now(UTC)
            if self.state.halted or self.tier.name == Tier.LOW:
                continue  # LOW holds overnight
            if calendar.in_eod_flatten_window(now) and self.state.positions:
                log.info("eod.flatten", n=len(self.state.positions))
                for sym in list(self.state.positions):
                    await self._exit_position(sym, "eod")

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.settings.heartbeat_interval_s)
            now = datetime.now(UTC)
            self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
            self.repo.update_state(heartbeat_ts=now, last_data_ts=self.state.last_data_ts,
                                   peak_equity=self.state.peak_equity)
            self.repo.add_equity_snapshot(now, self.state.equity, self.state.cash,
                                          self.state.gross_exposure)
            await self.pubsub.publish("equity", {
                "ts": now.isoformat(), "equity": self.state.equity,
                "cash": self.state.cash, "gross": self.state.gross_exposure})
            await self.pubsub.publish("positions", {
                "positions": [{"symbol": p.symbol, "qty": p.qty, "entry": p.entry_price,
                               "mark": p.mark, "upnl": p.unrealized_pnl,
                               "stop": p.stop_price}
                              for p in self.state.positions.values()]})

    async def command_poller(self) -> None:
        """API -> engine control channel (kill / reset / set_tier)."""
        while True:
            await asyncio.sleep(2)
            for cmd in self.repo.pending_commands():
                log.info("command.received", command=cmd.command)
                if cmd.command == "kill":
                    await self.kill.fire("manual kill via API")
                elif cmd.command == "reset":
                    self.state.halted = False
                    self.state.halted_reason = ""
                    self.repo.update_state(status="RUNNING", halted_reason="")
                    await self.pubsub.publish("engine_status", {"status": "RUNNING"})
                elif cmd.command == "set_tier":
                    tier_name = str(cmd.payload_json.get("tier", "")).lower()
                    if tier_name in [t.value for t in Tier] and not self.state.halted:
                        self.tier = TIERS[Tier(tier_name)]
                        self.risk.tier = self.tier
                        self.kill.tier = self.tier
                        self.repo.update_state(tier=tier_name)
                self.repo.mark_command_done(cmd.id)

    async def day_roll(self) -> None:
        """Reset day-start equity at each session open."""
        last_day = None
        while True:
            await asyncio.sleep(30)
            now = datetime.now(UTC)
            if calendar.is_session_open(now) and last_day != now.date():
                last_day = now.date()
                self.state.day_start_equity = self.state.equity
                self.repo.update_state(day_start_equity=self.state.equity)
                # shadow evaluation: refresh per-signal health multipliers
                self.ensemble.health_multipliers = self.health.evaluate(now)
                log.info("day.roll", equity=self.state.equity,
                         health=self.ensemble.health_multipliers)


def pubsub_emit(ps: PubSub) -> Any:
    async def emit(channel: str, data: dict[str, Any]) -> None:
        await ps.publish(channel, data)
    return emit
