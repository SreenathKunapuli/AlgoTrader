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
import time
from collections import defaultdict, deque
from datetime import UTC, date, datetime
from typing import Any, Literal

import numpy as np
import structlog

from .config.settings import Settings
from .config.tiers import TIERS, Tier, TierConfig
from .config.xsec import XSEC_BY_TIER
from .data import calendar
from .data.bar_builder import Bar, BarBuilder, aggregate
from .execution.order_manager import OrderManager
from .persistence.repo import Repo
from .pubsub import PubSub
from .risk.kill_switch import KillSwitch
from .risk.risk_manager import OrderIntent, Rejection, RiskManager
from .risk.sizing import size_position
from .risk.state import PortfolioState, Position
from .signals.ensemble import Ensemble
from .signals.health import SignalHealthTracker

log = structlog.get_logger()

ATR_PERIOD = 14
MAX_5M_BARS = 2400  # ~1 month of 5-min bars kept in memory per symbol
EXIT_INFLIGHT_S = 90.0  # one exit per symbol in flight; covers the 30s+30s
#                         limit->market exit ladder with margin


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
        self.risk = RiskManager(tier, state, xsec_cfg=XSEC_BY_TIER[tier.name])
        self.kill = KillSwitch(state, tier, repo, order_manager, pubsub_emit(pubsub),
                               staleness_kill_s=settings.staleness_kill_s,
                               broker_error_count=settings.broker_error_kill_count,
                               broker_error_window_s=settings.broker_error_kill_window_s,
                               flatten_intraday=self._flatten_intraday)
        self.builder = BarBuilder(interval_s=settings.bar_interval_s)
        self.bars_1m: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=12000))
        self.bars_5m: dict[str, deque[Bar]] = defaultdict(lambda: deque(maxlen=MAX_5M_BARS))
        self._pending_1m: dict[str, list[Bar]] = defaultdict(list)
        # microstructure accumulators drained into each incoming stream bar
        self._spread_acc: dict[str, list[float]] = defaultdict(list)
        self._qimb_acc: dict[str, list[float]] = defaultdict(list)
        self._flow_acc: dict[str, float] = defaultdict(float)
        self._flow_tot: dict[str, float] = defaultdict(float)
        self._last_px: dict[str, float] = {}
        self._last_quote: dict[str, tuple[float, float]] = {}
        self._last_rebalance = datetime.min.replace(tzinfo=UTC)
        self.health = SignalHealthTracker(repo, list(ensemble.signals))
        self._paused_stale = False
        # set by cli when the xsec book is enabled; guards below keep the
        # intraday paths off xsec positions even before the retag loop runs
        self.xsec: Any = None
        # COIDs of intraday add-on orders placed on top of xsec positions;
        # used by the fill handler to route these fills as "intraday" rather than "xsec"
        self._xsec_intraday_coids: set[str] = set()
        # starts as the frozen tier universe; day_scanner appends to this each session
        self._live_universe: list[str] = list(tier.universe)
        # stop prices staged at order-submit time, applied on fill (avoids async race)
        self._pending_stops: dict[str, float] = {}
        # monotonic ts of the last exit submitted per symbol — stop/signal/EOD
        # checks re-fire faster than the exit ladder fills; a second exit order
        # filling alongside the first would flip the position
        self._inflight_exits: dict[str, float] = {}
        # date of the last day_roll reset; day_scanner waits on this so the
        # morning scan is never wiped by the roll that follows session open
        self.day_rolled_date: date | None = None

    @staticmethod
    def _xsec_base_qty(pos: Position) -> int:
        """Shares owned by the xsec book. xsec_qty == 0 on an xsec-tagged
        position means retag hasn't run yet — treat it all as base."""
        return pos.xsec_qty if pos.xsec_qty > 0 else pos.qty

    def _is_xsec(self, symbol: str) -> bool:
        pos = self.state.positions.get(symbol)
        if pos is not None and pos.book == "xsec":
            return True
        return bool(self.xsec and symbol in self.xsec.holdings)

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
        bid, ask = self._last_quote.get(symbol, (0.0, 0.0))
        last = self._last_px.get(symbol)
        sign = 0
        if ask > 0 and price >= ask:
            sign = 1
        elif bid > 0 and price <= bid:
            sign = -1
        elif last is not None and price != last:
            sign = 1 if price > last else -1
        self._flow_acc[symbol] += sign * size
        self._flow_tot[symbol] += size
        self._last_px[symbol] = price
        pos = self.state.positions.get(symbol)
        if pos:
            pos.mark = price

    async def on_quote(self, symbol: str, ts: datetime, bid: float, bid_sz: int,
                       ask: float, ask_sz: int) -> None:
        self.state.last_data_ts = datetime.now(UTC)
        self._last_quote[symbol] = (bid, ask)
        if bid > 0 and ask > bid:
            self._spread_acc[symbol].append(ask - bid)
        denom = bid_sz + ask_sz
        if denom > 0:
            self._qimb_acc[symbol].append((bid_sz - ask_sz) / denom)

    async def on_stream_bar(self, symbol: str, ts: datetime, o: float, h: float,
                            low: float, c: float, vol: int, vwap: float,
                            tcount: int) -> None:
        """Official 1-min bar from Alpaca, enriched with accumulated
        quote/trade microstructure (zeros where we hold no subscription)."""
        self.state.last_data_ts = datetime.now(UTC)
        spreads = self._spread_acc.pop(symbol, [])
        qimbs = self._qimb_acc.pop(symbol, [])
        flow = self._flow_acc.pop(symbol, 0.0)
        tot = self._flow_tot.pop(symbol, 0.0)
        bar = Bar(
            symbol=symbol, ts=ts, interval_s=60, open=o, high=h, low=low,
            close=c, volume=vol, vwap=vwap, trade_count=tcount,
            mean_spread=sum(spreads) / len(spreads) if spreads else 0.0,
            mean_quote_imbalance=sum(qimbs) / len(qimbs) if qimbs else 0.0,
            flow_imbalance=flow / tot if tot > 0 else 0.0,
        )
        pos = self.state.positions.get(symbol)
        if pos:
            pos.mark = c
        await self.on_minute_bar(bar)

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
        trig = self.kill.check_triggers()
        if trig:
            scope, reason = trig
            if scope == "account" and not self.state.halted:
                await self.kill.fire(reason)
                return
            if scope == "intraday" and not self.state.intraday_halted:
                await self.kill.fire_intraday(reason)
        now = datetime.now(UTC)
        if (now - self._last_rebalance).total_seconds() >= self.tier.rebalance_seconds:
            self._last_rebalance = now
            await self.rebalance(now)

    # ---------------- decision path ---------------- #
    async def rebalance(self, now: datetime) -> None:
        if self.state.halted or self.state.intraday_halted or self._paused_stale:
            return
        if not calendar.in_entry_window(now):
            return

        # Advance trailing stops before computing new signals. Stops only ever
        # tighten (longs: up; shorts: down). A missing stop on an intraday
        # position means it predates this process (adopted by a reconcile) or
        # the fill event was missed — initialise it here so every intraday
        # position is protected. The xsec BASE stays stop-free by design
        # (monthly horizon; the account floor is its backstop) — only the
        # intraday addon layer (pos.qty > base) carries a stop.
        mult = self.tier.stop_atr_multiple
        for sym, pos in list(self.state.positions.items()):
            if pos.qty == 0:
                continue
            sym_bars = list(self.bars_5m.get(sym, ()))
            if len(sym_bars) < 30:
                continue
            atr = atr_from_bars(sym_bars)
            price = sym_bars[-1].close
            if pos.qty > 0:
                trailing = price - mult * atr
                if pos.book == "intraday":
                    if pos.stop_price is None or trailing > pos.stop_price:
                        pos.stop_price = trailing
                        log.info("stop.trail", symbol=sym,
                                 stop=round(trailing, 2), price=round(price, 2))
                elif pos.book == "xsec" and pos.qty > self._xsec_base_qty(pos):
                    if pos.stop_price is None or trailing > pos.stop_price:
                        pos.stop_price = trailing
                        log.info("stop.trail_addon", symbol=sym, stop=round(trailing, 2))
            elif pos.book == "intraday":  # short: trail the stop downward
                trailing = price + mult * atr
                if pos.stop_price is None or trailing < pos.stop_price:
                    pos.stop_price = trailing
                    log.info("stop.trail", symbol=sym,
                             stop=round(trailing, 2), price=round(price, 2))

        for symbol in self._live_universe:
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
        """Exit a position when the ensemble is no longer confident enough to hold it.

        Called only from the `not is_candidate` branch of rebalance(), so score is
        already below the confidence threshold. We exit on that alone — a weakening
        but still-positive signal (e.g. 0.3 with threshold 0.45) is not worth holding
        because the same gate that blocked entry now blocks continued holding.
        """
        pos = self.state.positions.get(symbol)
        if not pos:
            return
        threshold = self.tier.confidence_threshold
        if pos.book == "intraday":
            # Exit longs when not confidently bullish; shorts when not confidently bearish
            if (pos.qty > 0 and score < threshold) or (pos.qty < 0 and score > -threshold):
                await self._exit_position(symbol, "signal")
        elif pos.book == "xsec":
            # Signal death closes only the intraday ADDON layer. The base is a
            # monthly 12-1 momentum bet — a 5-min ensemble score is not evidence
            # against it, and exiting it here re-buys at the next rebalance for
            # pure spread cost. The base exits at rebalance, its book breaker,
            # or the account floor.
            base = self._xsec_base_qty(pos)
            if pos.qty > base and score < threshold:
                log.info("signal.xsec_addon_exit", symbol=symbol, score=round(score, 3))
                await self._exit_position(symbol, "signal", target_qty=base)

    async def _enter_or_adjust(self, symbol: str, bars: list[Bar],
                               res: Any, now: datetime) -> None:
        is_xsec = self._is_xsec(symbol)
        if is_xsec and res.final_score < 0:
            return  # never take an intraday short against an xsec long position

        price = bars[-1].close
        atr = atr_from_bars(bars)
        target_qty = size_position(self.tier, self.state.equity, price, atr, res.vol_mult)
        if res.final_score < 0:
            target_qty = -target_qty if self.tier.allow_short else 0

        pos = self.state.positions.get(symbol)
        # For xsec symbols: treat the intraday add-on layer as cur_qty (not the full
        # position, which includes the monthly base the xsec book owns).
        cur_qty = ((pos.qty - pos.xsec_qty) if pos else 0) if is_xsec else (pos.qty if pos else 0)

        delta = target_qty - cur_qty
        if delta == 0 or (target_qty == 0 and cur_qty == 0):
            return
        # Dead-band: skip drift adjustments below 10% of the target — a 1-2
        # share rebalance pays the spread for no edge. Full exits (target 0)
        # and direction flips pass through.
        if (cur_qty != 0 and target_qty != 0
                and (cur_qty > 0) == (target_qty > 0)
                and abs(delta) < 0.10 * abs(target_qty)):
            return
        side: Literal["buy", "sell"] = "buy" if delta > 0 else "sell"

        is_new_entry = cur_qty == 0 or (cur_qty > 0 and delta > 0) or (cur_qty < 0 and delta < 0)

        # LOB flow directional gate — only for new entries, not exits/adjustments.
        # Only applied when confidence >= lob_flow_min_conf (below that the score is
        # noise from sparse book data and should not veto the other signals).
        # One-sided: blocks when flow opposes direction; does NOT require positive flow
        # to buy (73% of buy candidates have negative LOB even on valid trades — a
        # symmetric gate would shut the engine down).
        if is_new_entry:
            lob = res.per_signal.get("lob_flow", {})
            lob_score = lob.get("score", 0.0)
            lob_conf  = lob.get("confidence", 0.0)
            gate      = self.tier.lob_flow_gate
            min_conf  = self.tier.lob_flow_min_conf
            if lob_conf >= min_conf:
                if delta > 0 and lob_score < -gate:
                    log.info("lob_flow.gate_blocked", symbol=symbol,
                             lob_score=round(lob_score, 3), lob_conf=round(lob_conf, 3),
                             gate=gate, side="long")
                    self.repo.add_rejection(
                        f"lob_flow gate: score {lob_score:.3f} "
                        f"(conf {lob_conf:.2f}) < -{gate} for long",
                        {"symbol": symbol, "side": side,
                         "lob_score": lob_score, "lob_conf": lob_conf})
                    return
                if delta < 0 and lob_score > gate:
                    log.info("lob_flow.gate_blocked", symbol=symbol,
                             lob_score=round(lob_score, 3), lob_conf=round(lob_conf, 3),
                             gate=gate, side="short")
                    self.repo.add_rejection(
                        f"lob_flow gate: score {lob_score:.3f} "
                        f"(conf {lob_conf:.2f}) > {gate} for short",
                        {"symbol": symbol, "side": side,
                         "lob_score": lob_score, "lob_conf": lob_conf})
                    return

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
                await self.kill.fire(
                    f"{self.settings.broker_error_kill_count} broker errors in "
                    f"{self.settings.broker_error_kill_window_s}s")
            return
        if order:
            if is_xsec:
                # Track all COID variants so fills route correctly as "intraday"
                # regardless of whether the entry, repeg, or market-replace fires.
                coid = order.client_order_id
                self._xsec_intraday_coids.update({
                    coid, coid[:24] + "-rp", coid[:24] + "-mkt"})
                log.info("xsec.intraday_addon", symbol=symbol, qty=abs(delta))
            elif is_new_entry:
                # Stop side follows the RESULTING position, not the order side
                # (a scale-down sell on a long must never get a short-style stop
                # above price — that stops the remainder out on the next bar).
                # Add-ons tighten-only: never loosen an already-trailed stop.
                new_qty = cur_qty + delta
                stop_mult = self.tier.stop_atr_multiple
                stop = price - stop_mult * atr if new_qty > 0 else price + stop_mult * atr
                p = self.state.positions.get(symbol)
                if p is not None and p.stop_price is not None:
                    stop = (max(stop, p.stop_price) if new_qty > 0
                            else min(stop, p.stop_price))
                if p is not None:
                    p.stop_price = stop
                self._pending_stops[symbol] = stop
            await self.pubsub.publish("orders", {
                "symbol": symbol, "side": side, "qty": abs(delta),
                "ts": now.isoformat(), "reason": "signal"})

    async def _exit_position(self, symbol: str, reason: str, target_qty: int = 0) -> None:
        """Exit to `target_qty` (default 0 = full close). Positive target keeps that many
        shares — used by EOD flattener and signal-flip to close only the intraday addon
        on top of an xsec base while leaving the monthly position intact."""
        pos = self.state.positions.get(symbol)
        if not pos or pos.qty == 0:
            return
        close_qty = pos.qty - target_qty
        if close_qty == 0:
            return
        # One exit in flight per symbol: stop checks (every 5-min bar) and the
        # EOD flattener (every 20s) re-fire before the ~60s exit ladder fills;
        # stacked exit orders both filling flips the position.
        now_m = time.monotonic()
        last = self._inflight_exits.get(symbol)
        if last is not None and now_m - last < EXIT_INFLIGHT_S:
            return
        side: Literal["buy", "sell"] = "sell" if close_qty > 0 else "buy"
        intent = OrderIntent(symbol=symbol, side=side, qty=abs(close_qty),
                             price_hint=pos.mark, reason=reason)
        approval = self.risk.approve_exit(intent)
        if isinstance(approval, Rejection):
            self.repo.add_rejection(approval.reason, intent.as_dict())
            return
        try:
            await self.om.submit(approval, datetime.now(UTC),
                                 mid=pos.mark, spread=0.0)
            self._inflight_exits[symbol] = now_m
        except Exception as exc:
            log.error("exit.submit_failed", symbol=symbol, error=str(exc))

    async def _flatten_intraday(self, reason: str) -> None:
        """Graceful exit (urgent limit -> market) of all intraday exposure:
        every intraday-book position plus any intraday addon layer sitting on
        an xsec base. The xsec base is untouched. Kill switch's intraday path."""
        for sym, pos in list(self.state.positions.items()):
            if self._is_xsec(sym):
                base = self._xsec_base_qty(pos)
                if pos.qty > base:
                    await self._exit_position(sym, "kill", target_qty=base)
            else:
                await self._exit_position(sym, "kill")

    async def check_stops(self, bar: Bar) -> None:
        pos = self.state.positions.get(bar.symbol)
        if not pos or pos.stop_price is None:
            return
        hit = (pos.qty > 0 and bar.low <= pos.stop_price) or \
              (pos.qty < 0 and bar.high >= pos.stop_price)
        if hit:
            log.info("stop.hit", symbol=bar.symbol, stop=pos.stop_price)
            if pos.book == "xsec":
                # Close only the intraday addon; clear stop so it doesn't re-fire
                # on the next bar while the exit order is still pending.
                await self._exit_position(bar.symbol, "stop", target_qty=pos.xsec_qty)
                pos.stop_price = None
            else:
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
                # intraday scope: the xsec book doesn't depend on the stream
                await self.kill.fire_intraday(f"data staleness {stale:.0f}s")
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
            if self.state.halted:
                continue
            if not calendar.in_eod_flatten_window(now):
                continue
            open_positions = [(s, p) for s, p in self.state.positions.items()
                              if p.qty != 0]
            if not open_positions:
                continue
            # Day-trade discipline applies to the intraday book and to intraday
            # addons on xsec symbols. The xsec BASE holds overnight — flattening
            # a monthly momentum book at every close pays two spreads a month
            # for a 25-minute hold and forfeits the edge entirely.
            log.info("eod.flatten", n=len(open_positions))
            for sym, pos in open_positions:
                if pos.book == "xsec":
                    base = self._xsec_base_qty(pos)
                    if pos.qty > base:
                        await self._exit_position(sym, "eod", target_qty=base)
                else:
                    await self._exit_position(sym, "eod")

    async def heartbeat(self) -> None:
        last_beat = datetime.now(UTC)
        while True:
            await asyncio.sleep(self.settings.heartbeat_interval_s)
            now = datetime.now(UTC)
            gap = (now - last_beat).total_seconds()
            last_beat = now
            if gap > 120:  # machine slept: local mirror is suspect
                log.warning("wake.detected", slept_s=int(gap))
                try:
                    await self.om.reconcile_state()
                    if self.xsec is not None:  # restore book tags immediately —
                        self.xsec.retag()      # adopted positions default to intraday
                except Exception as exc:
                    log.error("wake.reconcile_failed", error=str(exc))
            self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
            def _pos_dict(p: Position, now: datetime = now) -> dict[str, Any]:
                return {
                    "symbol": p.symbol,
                    "side": "long" if p.qty > 0 else "short",
                    "qty": p.qty,
                    "book": p.book,
                    "entry": p.entry_price,
                    "mark": p.mark,
                    "market_value": round(p.market_value, 2),
                    "upnl": round(p.unrealized_pnl, 2),
                    "stop": p.stop_price,
                    "age_s": (now - p.entry_ts).total_seconds() if p.entry_ts else None,
                    "entry_signals": p.entry_signals,
                }
            pos_payload = [_pos_dict(p) for p in self.state.positions.values()]
            self.repo.update_state(heartbeat_ts=now, last_data_ts=self.state.last_data_ts,
                                   peak_equity=self.state.peak_equity,
                                   positions_json=pos_payload)
            self.repo.add_equity_snapshot(now, self.state.equity, self.state.cash,
                                          self.state.gross_exposure)
            await self.pubsub.publish("equity", {
                "ts": now.isoformat(), "equity": self.state.equity,
                "cash": self.state.cash, "gross": self.state.gross_exposure})
            await self.pubsub.publish("positions", {"positions": pos_payload})

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
                elif cmd.command == "reset_intraday":
                    self.state.intraday_halted = False
                    self.state.last_data_ts = datetime.now(UTC)
                    self.repo.update_state(status="RUNNING", halted_reason="")
                    await self.pubsub.publish("engine_status", {"status": "RUNNING"})
                elif cmd.command == "set_tier":
                    tier_name = str(cmd.payload_json.get("tier", "")).lower()
                    if tier_name in [t.value for t in Tier] and not self.state.halted:
                        self.tier = TIERS[Tier(tier_name)]
                        self.risk.tier = self.tier
                        self.risk.xsec_cfg = XSEC_BY_TIER[self.tier.name]
                        self.kill.tier = self.tier
                        if self.xsec is not None:  # profile follows the tier
                            self.xsec.cfg = self.risk.xsec_cfg
                        self.repo.update_state(tier=tier_name)
                self.repo.mark_command_done(cmd.id)

    def expand_universe(self, symbols: list[str]) -> None:
        """Add screener-sourced symbols to the live universe for this session.

        Caller (cli.day_scanner) must have already warmed up bar history via
        engine.warmup() before calling this so rebalance() has enough bars to
        score them immediately.
        """
        new = [s for s in symbols if s not in self._live_universe]
        if not new:
            return
        self._live_universe.extend(new)
        self.risk.add_to_universe(new)
        log.info("universe.expanded", added=new, total=len(self._live_universe))

    def rotate_universe(self, add: list[str], evict: list[str]) -> None:
        """Swap stale idle candidates out and fresh ones in.

        Only evicts symbols that have no active position and are not in the
        hardcoded tier universe — held positions are never touched (we need
        to keep monitoring them for exits).
        """
        protected = (set(self.tier.universe)
                     | set(self.state.positions)
                     | (set(self.xsec.holdings) if self.xsec else set()))
        actual_evict = [s for s in evict if s not in protected]
        if actual_evict:
            self._live_universe = [s for s in self._live_universe
                                   if s not in actual_evict]
            for s in actual_evict:
                self.risk._dynamic_universe.discard(s)
            log.info("universe.evicted", removed=actual_evict,
                     total=len(self._live_universe))
        new = [s for s in add if s not in self._live_universe]
        if new:
            self._live_universe.extend(new)
            self.risk.add_to_universe(new)
            log.info("universe.rotated_in", added=new, total=len(self._live_universe))

    async def day_roll(self) -> None:
        """Reset day-start equity at each session open."""
        last_day = None
        while True:
            await asyncio.sleep(30)
            now = datetime.now(UTC)
            if calendar.is_session_open(now) and last_day != now.date():
                last_day = now.date()
                self.day_rolled_date = last_day
                self.state.day_start_equity = self.state.equity
                self.state.intraday_realized_today = 0.0
                if self.state.intraday_halted:  # day-scoped halt: new-day amnesty
                    self.state.intraday_halted = False
                    self.repo.update_state(status="RUNNING", halted_reason="")
                    log.info("intraday_halt.cleared")
                self.repo.update_state(day_start_equity=self.state.equity)
                # Reset dynamic universe — yesterday's movers don't carry over.
                # But symbols we still HOLD (and the xsec book) must stay in the
                # universe or their bars stop flowing and stops/exits go blind.
                keep = set(self.state.positions)
                if self.xsec is not None:
                    keep |= set(self.xsec.holdings)
                extra = [s for s in keep if s not in self.tier.universe]
                self._live_universe = list(self.tier.universe) + extra
                self.risk.reset_dynamic_universe()
                if extra:
                    self.risk.add_to_universe(extra)
                # shadow evaluation: refresh per-signal health multipliers
                self.ensemble.health_multipliers = self.health.evaluate(now)
                log.info("day.roll", equity=self.state.equity,
                         health=self.ensemble.health_multipliers)


def pubsub_emit(ps: PubSub) -> Any:
    async def emit(channel: str, data: dict[str, Any]) -> None:
        await ps.publish(channel, data)
    return emit
