"""lobctl — engine entrypoint: run / halt / reset / status / flatten / train-signal.

`run` wires real Alpaca clients; everything else talks through the DB
(commands table / engine_state), so it works whether or not the engine
process is up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

import structlog

from .config.settings import get_settings

if TYPE_CHECKING:
    from .config.settings import Settings
    from .config.tiers import TierConfig
    from .data.alpaca_stream import MarketStream
    from .engine import Engine
    from .persistence.repo import Repo
from .config.tiers import TIERS, Tier

structlog.configure(processors=[
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.add_log_level,
    structlog.processors.JSONRenderer(),
])
log = structlog.get_logger()


def _repo() -> Repo:
    from .persistence.repo import Repo

    s = get_settings()
    return Repo(s.resolved_database_url())


async def _day_scanner(engine: Engine, stream: MarketStream,
                       settings: Settings, tier: TierConfig) -> None:
    """Continuously refresh the trading universe throughout the day.

    Phase 1 — morning scan (once per calendar day, fires at engine startup):
        Scores ~3 000 stocks by short-term daily momentum (universe3000.csv →
        sp500_constituents.csv fallback), warms up minute-bar history for the top
        candidates, and seeds the live universe before the first bar arrives.
        Reserves 8 stream slots for intraday screener picks.

    Phase 2 — intraday rescan (every 30 min during market hours):
        Calls the Alpaca screener for today's top gainers and most-active names.
        If new candidates appear that aren't already subscribed, idle (no-position)
        members of the universe are evicted in FIFO order to free slots, then new
        names are warmed up and added. This catches underdogs that break out mid-day
        on news, earnings, or sudden volume surges — not just pre-market leaders.

    Stream budget: SUBSCRIPTION_LIMIT = 30.
    Protected from eviction: tier base universe, any open position symbols.
    """
    import datetime as dt

    from .data import calendar
    from .data.alpaca_stream import SUBSCRIPTION_LIMIT
    from .data.history import fetch_minute_bars
    from .data.morning_scan import build_daily_candidates
    from .data.screener import scan_candidates

    INTRADAY_RESCAN_S = 1800          # rescan screener every 30 min
    INTRADAY_SLOT_RESERVE = 8         # always keep 8 slots open for intraday picks

    morning_done: set[dt.date] = set()  # dates where morning scan completed
    last_intraday_ts = dt.datetime.min.replace(tzinfo=dt.UTC)
    dynamic_added: list[str] = []     # FIFO queue of dynamically-added symbols

    while True:
        await asyncio.sleep(30)
        now = dt.datetime.now(dt.UTC)
        today = now.date()

        # ── Phase 1: morning scan ────────────────────────────────────────────
        # Wait for the session to open AND for the engine's day_roll to have
        # reset the universe. Running earlier (the loop wakes at midnight UTC
        # when the calendar date flips) gets its additions wiped by that reset.
        if (today not in morning_done
                and calendar.is_session_open(now)
                and engine.day_rolled_date == today):
            morning_done.add(today)
            dynamic_added = []   # reset FIFO for the new session

            exclude = set(engine._live_universe)
            if engine.xsec:
                exclude |= set(engine.xsec.holdings)
            slots = max(0, SUBSCRIPTION_LIMIT - len(engine._live_universe)
                        - INTRADAY_SLOT_RESERVE)
            if slots > 0:
                try:
                    candidates = await asyncio.to_thread(
                        build_daily_candidates,
                        settings.alpaca_api_key, settings.alpaca_secret_key,
                        top_n=slots, exclude=exclude,
                    )
                except Exception as exc:
                    log.warning("day_scanner.morning_failed", error=str(exc))
                    candidates = []
                if candidates:
                    try:
                        history = await asyncio.to_thread(
                            fetch_minute_bars,
                            settings.alpaca_api_key, settings.alpaca_secret_key,
                            candidates, settings.history_warmup_days,
                        )
                        engine.warmup(history)
                    except Exception as exc:
                        log.warning("day_scanner.morning_warmup_failed", error=str(exc))
                    engine.expand_universe(candidates)
                    dynamic_added.extend(candidates)
                    engine.state.last_data_ts = dt.datetime.now(dt.UTC)
                    stream.update_symbols(engine._live_universe)
                    log.info("day_scanner.morning_done", added=len(candidates),
                             top5=candidates[:5],
                             universe=len(engine._live_universe))

        # ── Phase 2: intraday rescan ─────────────────────────────────────────
        if not calendar.is_session_open(now):
            continue
        if (now - last_intraday_ts).total_seconds() < INTRADAY_RESCAN_S:
            continue
        last_intraday_ts = now

        exclude = set(engine._live_universe)
        try:
            new_candidates = await asyncio.to_thread(
                scan_candidates,
                settings.alpaca_api_key, settings.alpaca_secret_key,
                exclude, 10,
            )
        except Exception as exc:
            log.warning("day_scanner.intraday_failed", error=str(exc))
            continue

        if not new_candidates:
            log.info("day_scanner.intraday_no_new")
            continue

        # How many slots do we need to free?
        slots_free = SUBSCRIPTION_LIMIT - len(engine._live_universe)
        slots_needed = max(0, len(new_candidates) - slots_free)

        # Evict oldest idle (no-position) dynamic names to make room
        held = set(engine.state.positions)
        evict: list[str] = []
        if slots_needed > 0:
            for sym in list(dynamic_added):
                if sym not in held and sym not in tier.universe:
                    evict.append(sym)
                    if len(evict) >= slots_needed:
                        break

        engine.rotate_universe(new_candidates, evict)

        for sym in evict:
            if sym in dynamic_added:
                dynamic_added.remove(sym)
        dynamic_added.extend(new_candidates)

        # Warm up history only for symbols that were actually added and lack bars
        actually_added = [s for s in new_candidates
                          if s in engine._live_universe
                          and len(engine.bars_5m.get(s, [])) < 30]
        if actually_added:
            try:
                history = await asyncio.to_thread(
                    fetch_minute_bars,
                    settings.alpaca_api_key, settings.alpaca_secret_key,
                    actually_added, settings.history_warmup_days,
                )
                engine.warmup(history)
            except Exception as exc:
                log.warning("day_scanner.intraday_warmup_failed", error=str(exc))

        # Reset staleness clock before reconnect so the monitor doesn't fire
        # during the few seconds the stream is tearing down and rebuilding.
        engine.state.last_data_ts = dt.datetime.now(dt.UTC)
        stream.update_symbols(engine._live_universe)
        log.info("day_scanner.intraday_done",
                 added=new_candidates, evicted=evict,
                 universe=len(engine._live_universe))


async def _run(tier_name: str) -> None:
    import atexit
    import os
    from pathlib import Path

    from .data.alpaca_stream import MarketStream, TradeUpdateStream
    from .data.history import fetch_minute_bars
    from .engine import Engine
    from .execution.broker import AlpacaBroker
    from .execution.order_manager import OrderManager
    from .execution.reconcile import reconcile
    from .persistence.repo import Repo
    from .pubsub import PubSub
    from .risk.state import PortfolioState
    from .signals.ensemble import Ensemble
    from .signals.lob_flow import LobFlowSignal
    from .signals.mean_reversion import MeanReversionSignal
    from .signals.momentum import MomentumSignal

    # Alpaca allows ONE data websocket per account: two engines silently kick
    # each other off the stream (observed live 2026-07-07). Refuse dual launch.
    lock = Path("lobengine.pid")
    if lock.exists():
        try:
            old_pid = int(lock.read_text().strip())
            os.kill(old_pid, 0)  # raises if not running
            print(f"FATAL: another engine is already running (pid {old_pid}). "
                  "Alpaca allows one data connection — two engines starve each other. "
                  "Stop it first (Ctrl-C or `kill`).")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # stale lockfile
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))

    s = get_settings()
    if s.trading_mode != "paper":
        print("FATAL: only TRADING_MODE=paper is supported in this build.")
        sys.exit(2)
    if not s.alpaca_api_key or not s.alpaca_secret_key:
        print("FATAL: ALPACA_API_KEY / ALPACA_SECRET_KEY missing in .env")
        sys.exit(2)

    tier = TIERS[Tier(tier_name)]
    repo = Repo(s.resolved_database_url())
    state = PortfolioState()
    db_state = repo.get_state()
    if db_state.status == "HALTED":
        print(f"Engine is HALTED ({db_state.halted_reason}). Run `lobctl reset` first.")
        sys.exit(1)
    state.peak_equity = db_state.peak_equity
    pubsub = PubSub(s.redis_url)
    broker = AlpacaBroker(s.alpaca_api_key, s.alpaca_secret_key, s.alpaca_paper_base_url)
    om = OrderManager(broker, repo, state)
    ensemble = Ensemble([MomentumSignal(), MeanReversionSignal(),
                         LobFlowSignal(s.models_dir)])
    engine = Engine(s, tier, repo, om, ensemble, state, pubsub)

    await reconcile(broker, repo, state)
    state.day_start_equity = state.equity
    state.peak_equity = max(state.peak_equity, state.equity)
    repo.update_state(status="RUNNING", tier=tier_name,
                      day_start_equity=state.equity, peak_equity=state.peak_equity)

    log.info("warmup.backfill", days=s.history_warmup_days)
    history = fetch_minute_bars(s.alpaca_api_key, s.alpaca_secret_key,
                                tier.universe, s.history_warmup_days)
    engine.warmup(history)

    stream = MarketStream(s.alpaca_api_key, s.alpaca_secret_key, tier.universe,
                          engine.on_trade, engine.on_quote, engine.on_stream_bar)

    async def _on_fill_event(symbol: str, side: str, qty: int, price: float,
                             coid: str) -> None:
        # Determine book: COID set identifies intraday add-ons on xsec symbols.
        # All three COID variants (entry, -rp repeg, -mkt market-replace) are tracked.
        if coid in engine._xsec_intraday_coids:
            engine._xsec_intraday_coids.discard(coid)
            book = "intraday"
        elif engine.xsec and symbol in engine.xsec.holdings:
            book = "xsec"
        else:
            book = "intraday"
        om.on_fill(symbol, side, qty, price, reason="stream", book=book)
        # Keep xsec_qty in sync with each xsec-book fill (incremental, not reset).
        if book == "xsec":
            pos = engine.state.positions.get(symbol)
            if pos:
                xsec_delta = qty if side == "buy" else -qty
                pos.xsec_qty = max(0, pos.xsec_qty + xsec_delta)
        # Apply ATR stop staged at submit time for non-xsec intraday fills.
        if book == "intraday":
            stop = engine._pending_stops.pop(symbol, None)
            if stop is not None:
                pos = engine.state.positions.get(symbol)
                if pos and pos.book == "intraday":
                    pos.stop_price = stop

    trade_stream = TradeUpdateStream(s.alpaca_api_key, s.alpaca_secret_key,
                                     paper=True, on_fill=_on_fill_event)
    tasks = [
        stream.run_forever(),
        trade_stream.run_forever(),
        engine.staleness_monitor(),
        engine.eod_flattener(),
        engine.heartbeat(),
        engine.command_poller(),
        engine.day_roll(),
        _day_scanner(engine, stream, s, tier),
    ]
    if s.xsec_enabled:
        from .config.xsec import XSEC_BY_TIER
        from .strategy.xsec_momentum import XsecMomentumStrategy

        xsec = XsecMomentumStrategy(s, state, engine.risk, om, repo, pubsub,
                                    cfg=XSEC_BY_TIER[tier.name])
        engine.xsec = xsec
        xsec.retag()  # restore book tags over the freshly reconciled mirror
        tasks.append(xsec.run())
        log.info("xsec.enabled", profile=tier_name, top_n=xsec.cfg.top_n,
                 holdings=len(xsec.holdings), pending=len(xsec.pending),
                 last_rebalance=xsec.last_rebalance_month)

        # Add xsec holdings to the intraday universe so 5-min signals are computed
        # and intraday long add-ons can fire when momentum confirms the monthly thesis.
        if xsec.holdings:
            xsec_syms = list(xsec.holdings)
            log.info("warmup.xsec_holdings", syms=xsec_syms)
            try:
                xsec_hist = fetch_minute_bars(s.alpaca_api_key, s.alpaca_secret_key,
                                              xsec_syms, s.history_warmup_days)
                engine.warmup(xsec_hist)
            except Exception as exc:
                log.warning("warmup.xsec_failed", error=str(exc))
            engine.expand_universe(xsec_syms)
            stream.update_symbols(engine._live_universe)
    log.info("engine.start", tier=tier_name, universe=len(tier.universe))
    await asyncio.gather(*tasks)


def main() -> None:
    p = argparse.ArgumentParser(prog="lobctl")
    sub = p.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run the engine (paper only)")
    runp.add_argument("--tier", default=get_settings().risk_tier,
                      choices=["low", "medium", "high"])
    sub.add_parser("status")
    sub.add_parser("halt", help="fire the kill switch")
    sub.add_parser("reset", help="clear HALTED state")
    sub.add_parser("flatten", help="alias for halt (cancel+flatten)")
    trainp = sub.add_parser("train-signal", help="walk-forward train lob_flow")
    trainp.add_argument("--days", type=int, default=30)
    evalp = sub.add_parser("evaluate-signal", help="§5.9 cost-aware walk-forward eval")
    evalp.add_argument("--days", type=int, default=30)
    args = p.parse_args()

    if args.cmd == "run":
        asyncio.run(_run(args.tier))
    elif args.cmd == "status":
        st = _repo().get_state()
        print(f"status={st.status} tier={st.tier} reason={st.halted_reason!r} "
              f"heartbeat={st.heartbeat_ts} peak={st.peak_equity}")
    elif args.cmd in ("halt", "flatten"):
        _repo().enqueue_command("kill")
        print("kill command enqueued (engine executes within 2s if running)")
    elif args.cmd == "reset":
        repo = _repo()
        repo.enqueue_command("reset")
        repo.update_state(status="STOPPED", halted_reason="")
        print("reset enqueued + state cleared")
    elif args.cmd == "train-signal":
        from .train_signal import run_training

        run_training(days=args.days)
    elif args.cmd == "evaluate-signal":
        from .data.history import fetch_minute_bars
        from .evaluate_signal import evaluate
        from .train_signal import to_5min

        s = get_settings()
        hist = fetch_minute_bars(s.alpaca_api_key, s.alpaca_secret_key,
                                 TIERS[Tier.MEDIUM].universe, args.days)
        bars5 = {sym: to_5min(b) for sym, b in hist.items()}
        m = evaluate(bars5, s.models_dir, arch=s.lob_flow_arch)
        print(f"gate={'PASS' if m.passes_gate else 'FAIL'} {m.as_dict()}")


if __name__ == "__main__":
    main()
