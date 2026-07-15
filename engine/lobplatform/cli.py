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


async def _day_scanner(engine: "Engine", stream: "MarketStream",
                       settings: "Settings", tier: "TierConfig") -> None:
    """Once per session: wait for open + 60s, scan top gainers/most-active,
    warm up their bar history, expand the engine's live universe, and trigger
    a stream reconnect so new symbols start receiving bars.

    Dynamic symbols are session-scoped: engine.day_roll() resets them each
    morning before the next scan fires.
    """
    from .data import calendar
    from .data.alpaca_stream import SUBSCRIPTION_LIMIT
    from .data.history import fetch_minute_bars
    from .data.screener import scan_candidates

    # One scan slot per session: open + 60s (gap plays, pre-market catalysts).
    # A midday rescan was considered but dropped: screener picks are gated at
    # 10:30am ET in _enter_or_adjust (volume collapses on small/mid-caps after
    # the opening hour), so any new names found later could not enter anyway.
    SCAN_OFFSETS_S = [60]   # seconds after session open

    session_open_ts: dict = {}   # date -> aware datetime of session open
    completed_slots: dict = {}   # date -> set of completed slot indices

    while True:
        await asyncio.sleep(30)
        dt = __import__("datetime")
        now = dt.datetime.now(dt.timezone.utc)
        if not calendar.is_session_open(now):
            continue
        today = now.date()

        if today not in session_open_ts:
            session_open_ts[today] = now   # approximate open time on first detection
        if today not in completed_slots:
            completed_slots[today] = set()

        open_ts = session_open_ts[today]
        elapsed = (now - open_ts).total_seconds()

        # find the next slot that's due and not yet completed
        slot_idx = None
        for i, offset in enumerate(SCAN_OFFSETS_S):
            if i not in completed_slots[today] and elapsed >= offset:
                slot_idx = i
                break
        if slot_idx is None:
            continue

        completed_slots[today].add(slot_idx)
        label = "open" if slot_idx == 0 else "midday"

        # only add symbols not already in the live universe
        exclude = set(engine._live_universe)
        if engine.xsec:
            exclude |= set(engine.xsec.holdings)
        max_n = max(0, SUBSCRIPTION_LIMIT - len(engine._live_universe))
        if max_n == 0:
            log.info("day_scanner.slots_full", slot=label)
            continue

        try:
            candidates = await asyncio.to_thread(
                scan_candidates, settings.alpaca_api_key, settings.alpaca_secret_key,
                exclude, max_n,
            )
        except Exception as exc:
            log.warning("day_scanner.scan_failed", slot=label, error=str(exc))
            continue

        if not candidates:
            log.info("day_scanner.no_candidates", slot=label)
            continue

        try:
            history = await asyncio.to_thread(
                fetch_minute_bars, settings.alpaca_api_key, settings.alpaca_secret_key,
                candidates, settings.history_warmup_days,
            )
            engine.warmup(history)
        except Exception as exc:
            log.warning("day_scanner.warmup_failed", slot=label, error=str(exc))
            continue

        engine.expand_universe(candidates)
        stream.update_symbols(engine._live_universe)
        log.info("day_scanner.done", slot=label, added=candidates,
                 universe_size=len(engine._live_universe))


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
