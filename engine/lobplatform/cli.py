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


async def _run(tier_name: str) -> None:
    from .data.alpaca_stream import MarketStream
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
    log.info("engine.start", tier=tier_name, universe=len(tier.universe))
    await asyncio.gather(
        stream.run_forever(),
        engine.staleness_monitor(),
        engine.eod_flattener(),
        engine.heartbeat(),
        engine.command_poller(),
        engine.day_roll(),
    )


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
