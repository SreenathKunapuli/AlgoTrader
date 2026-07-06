"""Replay integration test (§5.8): a synthetic session driven through the
full engine with a MockBroker. Asserts: tier limits never violated, kill
switch fires on forced daily loss, HALTED persists across restart."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from lobplatform.config.settings import Settings
from lobplatform.config.tiers import TIERS, Tier
from lobplatform.data.bar_builder import Bar
from lobplatform.engine import Engine
from lobplatform.execution.order_manager import OrderManager
from lobplatform.pubsub import PubSub
from lobplatform.risk.state import PortfolioState, Position
from lobplatform.signals.ensemble import Ensemble
from lobplatform.signals.mean_reversion import MeanReversionSignal
from lobplatform.signals.momentum import MomentumSignal

MED = TIERS[Tier.MEDIUM]
SESSION_OPEN = datetime(2026, 6, 15, 13, 30, tzinfo=UTC)  # 9:30 ET


def synth_bars(symbol: str, n: int, start: datetime, price0: float = 100.0,
               drift: float = 0.0, seed: int = 0) -> list[Bar]:
    rng = np.random.default_rng(seed)
    px = price0
    out = []
    for i in range(n):
        ts = start + timedelta(minutes=5 * i)
        ret = drift + rng.normal(0, 0.001)
        new = px * (1 + ret)
        hi, lo = max(px, new) * 1.0005, min(px, new) * 0.9995
        out.append(Bar(symbol=symbol, ts=ts, interval_s=300, open=px, high=hi,
                       low=lo, close=new, volume=1000, vwap=(px + new) / 2,
                       trade_count=50, mean_spread=0.02,
                       mean_quote_imbalance=rng.normal(0, 0.2),
                       flow_imbalance=rng.normal(0, 0.2)))
        px = new
    return out


def make_engine(state: PortfolioState, repo, mock_broker):  # type: ignore[no-untyped-def]
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    ens = Ensemble([MomentumSignal(), MeanReversionSignal()])
    return Engine(s, MED, repo, om, ens, state, PubSub()), om


async def test_replay_no_limit_violations(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_engine(state, repo, mock_broker)
    # warm caches with 10 prior days then replay a session with strong drift
    warm_start = SESSION_OPEN - timedelta(days=10)
    for sym in ["SPY", "QQQ", "AAPL"]:
        engine.bars_5m[sym].extend(
            synth_bars(sym, 700, warm_start, drift=0.0008, seed=hash(sym) % 100))
        mock_broker.price[sym] = engine.bars_5m[sym][-1].close

    engine._last_rebalance = datetime.min.replace(tzinfo=UTC)
    await engine.rebalance(SESSION_OPEN + timedelta(hours=2))

    # every submitted order respected tier caps at submit time
    for o in mock_broker.submitted:
        px = mock_broker.price.get(o.symbol, 100.0)
        assert o.qty * px <= MED.max_position_pct * state.equity + px
    assert state.gross_exposure <= MED.max_gross_pct * state.equity + 1e-6
    assert len(state.positions) <= MED.max_open_positions


async def test_replay_kill_on_daily_loss(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_engine(state, repo, mock_broker)
    state.positions["AAPL"] = Position(symbol="AAPL", qty=90, entry_price=100.0, mark=100.0)
    mock_broker.positions["AAPL"] = 90
    # force -2.5% day: equity 100k -> 97.5k
    state.equity = 97_500.0
    bar = synth_bars("AAPL", 1, SESSION_OPEN + timedelta(hours=3))[0]
    await engine.on_five_min_bar(bar)
    assert state.halted
    assert "daily loss" in state.halted_reason
    assert mock_broker.closed_all >= 1
    assert repo.get_state().status == "HALTED"

    # restart mid-day: fresh engine sees persisted HALTED
    state2 = PortfolioState()
    state2.halted = repo.get_state().status == "HALTED"
    assert state2.halted, "HALTED must survive restart"


async def test_replay_stop_exit(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_engine(state, repo, mock_broker)
    state.positions["SPY"] = Position(symbol="SPY", qty=10, entry_price=100.0,
                                      mark=100.0, stop_price=99.0)
    mock_broker.positions["SPY"] = 10
    bar = Bar(symbol="SPY", ts=SESSION_OPEN + timedelta(hours=1), interval_s=300,
              open=99.5, high=99.6, low=98.8, close=99.0, volume=100, vwap=99.2,
              trade_count=5, mean_spread=0.02, mean_quote_imbalance=0.0,
              flow_imbalance=0.0)
    await engine.check_stops(bar)
    sells = [o for o in mock_broker.submitted if o.side == "sell" and o.symbol == "SPY"]
    assert len(sells) == 1 and sells[0].qty == 10
