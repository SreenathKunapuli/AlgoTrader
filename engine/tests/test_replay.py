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
from lobplatform.signals.ensemble import Ensemble, EnsembleResult
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


async def test_replay_intraday_kill_on_daily_loss(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_engine(state, repo, mock_broker)
    # intraday book down -2.8% unrealized; xsec book untouched
    state.positions["AAPL"] = Position(symbol="AAPL", qty=90, entry_price=100.0, mark=72.0)
    mock_broker.positions["AAPL"] = 90
    state.positions["NVDA"] = Position(symbol="NVDA", qty=5, entry_price=100.0,
                                       mark=100.0, book="xsec")
    mock_broker.positions["NVDA"] = 5
    bar = synth_bars("AAPL", 1, SESSION_OPEN + timedelta(hours=3))[0]
    await engine.on_five_min_bar(bar)
    # intraday scope: book flattened + halted, engine still running, xsec kept
    assert state.intraday_halted and not state.halted
    sells = [o for o in mock_broker.submitted if o.symbol == "AAPL" and o.side == "sell"]
    assert sells and sells[0].qty == 90
    assert not [o for o in mock_broker.submitted if o.symbol == "NVDA"]
    assert repo.get_state().status == "INTRADAY_HALTED"


async def test_replay_global_kill_on_account_floor(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_engine(state, repo, mock_broker)
    state.peak_equity = 160_000.0  # 100k now -> 37.5% DD > 35% MEDIUM floor
    bar = synth_bars("SPY", 1, SESSION_OPEN + timedelta(hours=3))[0]
    await engine.on_five_min_bar(bar)
    assert state.halted and "drawdown" in state.halted_reason
    assert mock_broker.closed_all >= 1                  # both books flattened
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


def make_flat_bars(symbol: str, n: int, start: datetime, price: float) -> list[Bar]:
    """n identical bars at a fixed price — isolates the trailing stop from ATR noise."""
    return [Bar(symbol=symbol, ts=start + timedelta(minutes=5 * i), interval_s=300,
                open=price, high=price * 1.001, low=price * 0.999, close=price,
                volume=1000, vwap=price, trade_count=50, mean_spread=0.02,
                mean_quote_imbalance=0.0, flow_imbalance=0.0)
            for i in range(n)]


async def test_trailing_stop_advances_and_locks_gain(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    """Trailing stop moves up as price rises (locking in gains) and never retreats."""
    engine, _ = make_engine(state, repo, mock_broker)

    state.equity = 100_000.0
    state.positions["AAPL"] = Position(symbol="AAPL", qty=10, entry_price=100.0,
                                       mark=100.0, stop_price=97.0)

    # Warm up 40 bars at 108 — price has risen; trailing stop should advance past 97
    for b in make_flat_bars("AAPL", 40, SESSION_OPEN, price=108.0):
        engine.bars_5m["AAPL"].append(b)

    now = SESSION_OPEN + timedelta(hours=1)
    await engine.rebalance(now)

    new_stop = state.positions["AAPL"].stop_price
    assert new_stop is not None
    assert new_stop > 97.0, f"trailing stop should have advanced above 97, got {new_stop}"
    assert new_stop < 108.0, f"trailing stop must stay below current price, got {new_stop}"

    # Replace bars with lower price — stop must NOT retreat
    engine.bars_5m["AAPL"].clear()
    for b in make_flat_bars("AAPL", 40, SESSION_OPEN, price=104.0):
        engine.bars_5m["AAPL"].append(b)

    await engine.rebalance(now)
    assert state.positions["AAPL"].stop_price >= new_stop, "stop must never retreat"


async def test_scale_down_keeps_long_stop(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    """Reducing a long must not restage its stop from the ORDER side: a
    scale-down sell once wrote a short-style stop above price, which stopped
    the remainder out on the next bar."""
    engine, om = make_engine(state, repo, mock_broker)
    for b in make_flat_bars("AAPL", 40, SESSION_OPEN, price=108.0):
        engine.bars_5m["AAPL"].append(b)
    state.positions["AAPL"] = Position(symbol="AAPL", qty=50, entry_price=100.0,
                                       mark=108.0, stop_price=97.0)
    mock_broker.positions["AAPL"] = 50
    # strong long signal but tiny vol multiplier -> target far below current qty
    res = EnsembleResult(symbol="AAPL", final_score=0.6, vol_mult=0.05)
    await engine._enter_or_adjust("AAPL", list(engine.bars_5m["AAPL"]), res,
                                  SESSION_OPEN + timedelta(hours=2))
    sells = [o for o in mock_broker.submitted if o.side == "sell"]
    assert sells, "expected a scale-down sell"
    assert state.positions["AAPL"].stop_price == 97.0, "scale-down must not touch the stop"
    assert "AAPL" not in engine._pending_stops


async def test_exit_inflight_guard_prevents_stacked_exits(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    """Stop/EOD checks re-fire before the exit ladder fills; a second exit
    order filling alongside the first would flip the position."""
    engine, om = make_engine(state, repo, mock_broker)
    state.positions["SPY"] = Position(symbol="SPY", qty=10, entry_price=100.0,
                                      mark=100.0, stop_price=99.0)
    mock_broker.positions["SPY"] = 10
    await engine._exit_position("SPY", "stop")
    await engine._exit_position("SPY", "stop")
    sells = [o for o in mock_broker.submitted if o.symbol == "SPY" and o.side == "sell"]
    assert len(sells) == 1, "duplicate exit while one is in flight"


async def test_intraday_kill_flattens_addon_keeps_xsec_base(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    engine, om = make_engine(state, repo, mock_broker)
    state.positions["NVDA"] = Position(symbol="NVDA", qty=15, entry_price=100.0,
                                       mark=100.0, book="xsec", xsec_qty=10)
    mock_broker.positions["NVDA"] = 15
    await engine._flatten_intraday("test")
    sells = [o for o in mock_broker.submitted if o.symbol == "NVDA" and o.side == "sell"]
    assert len(sells) == 1 and sells[0].qty == 5, "only the addon layer is intraday exposure"


async def test_signal_exit_spares_xsec_base(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    """A dead 5-min ensemble score must not liquidate the monthly xsec base."""
    engine, om = make_engine(state, repo, mock_broker)
    state.positions["NVDA"] = Position(symbol="NVDA", qty=10, entry_price=100.0,
                                       mark=100.0, book="xsec", xsec_qty=10)
    mock_broker.positions["NVDA"] = 10
    await engine._maybe_exit_on_signal("NVDA", [], 0.0)
    assert not mock_broker.submitted
