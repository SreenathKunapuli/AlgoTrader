"""Kill-switch action sequence, order-id idempotency, bar builder."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from lobplatform.config.tiers import TIERS, Tier
from lobplatform.data.bar_builder import BarBuilder, aggregate
from lobplatform.execution.order_manager import OrderManager, make_client_order_id
from lobplatform.risk.kill_switch import KillSwitch
from lobplatform.risk.state import Position

MED = TIERS[Tier.MEDIUM]


# ---------- kill switch ---------- #
async def test_kill_sequence(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    events: list[tuple[str, dict]] = []

    async def emit(ch: str, data: dict) -> None:  # type: ignore[type-arg]
        events.append((ch, data))

    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    mock_broker.positions["AAPL"] = 10
    state.positions["AAPL"] = Position(symbol="AAPL", qty=10, entry_price=100, mark=100)
    ks = KillSwitch(state, MED, repo, om, emit)

    await ks.fire("test trigger")

    assert state.halted and state.halted_reason == "test trigger"
    assert repo.get_state().status == "HALTED"          # (1) persisted first
    assert mock_broker.closed_all >= 1                  # (3) flattened
    assert not mock_broker.positions                    # (4) verified flat
    assert events and events[0][0] == "engine_status"   # (5) emitted
    # idempotent
    await ks.fire("second")
    assert state.halted_reason == "test trigger"


async def test_kill_persists_across_restart(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    async def emit(ch: str, data: dict) -> None:  # type: ignore[type-arg]
        pass

    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    ks = KillSwitch(state, MED, repo, om, emit)
    await ks.fire("daily loss")
    # simulate restart: fresh read of engine_state
    assert repo.get_state().status == "HALTED"
    assert repo.get_state().halted_reason == "daily loss"


def test_kill_triggers(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    async def emit(ch: str, data: dict) -> None:  # type: ignore[type-arg]
        pass

    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    ks = KillSwitch(state, MED, repo, om, emit)
    in_session = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)

    state.last_data_ts = in_session
    assert ks.check_triggers(in_session) is None
    state.equity = 97_000.0  # -3% day
    assert "daily loss" in (ks.check_triggers(in_session) or "")
    state.equity = 100_000.0
    state.peak_equity = 125_000.0
    assert "drawdown" in (ks.check_triggers(in_session) or "")
    state.peak_equity = 100_000.0
    state.last_data_ts = in_session - timedelta(seconds=200)
    assert "staleness" in (ks.check_triggers(in_session) or "")


def test_broker_error_trigger(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    async def emit(ch: str, data: dict) -> None:  # type: ignore[type-arg]
        pass

    ks = KillSwitch(state, MED, repo,
                    OrderManager(mock_broker, repo, state), emit)  # type: ignore[arg-type]
    fired = [ks.record_broker_error() for _ in range(5)]
    assert fired[-1] is True and not any(fired[:-1])


# ---------- idempotent order ids ---------- #
def test_client_order_id_stable() -> None:
    ts = datetime(2026, 6, 15, 14, 30, tzinfo=UTC)
    a = make_client_order_id("ensemble", "AAPL", "buy", ts)
    b = make_client_order_id("ensemble", "AAPL", "buy", ts)
    assert a == b and len(a) == 32
    assert make_client_order_id("ensemble", "AAPL", "sell", ts) != a
    assert make_client_order_id("ensemble", "AAPL", "buy", ts + timedelta(minutes=5)) != a


async def test_duplicate_submit_suppressed(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    from lobplatform.risk.risk_manager import OrderIntent, RiskManager

    rm = RiskManager(MED, state)
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    ts = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)
    intent = OrderIntent(symbol="AAPL", side="buy", qty=5, price_hint=100.0)
    ap = rm.approve(intent, ts)
    o1 = await om.submit(ap, ts, mid=100.0, spread=0.02)  # type: ignore[arg-type]
    o2 = await om.submit(ap, ts, mid=100.0, spread=0.02)  # type: ignore[arg-type]
    assert o1 is not None and o2 is None  # duplicate swallowed


# ---------- bar builder ---------- #
def test_bar_builder_ohlcv_and_flow() -> None:
    bb = BarBuilder(interval_s=60)
    t0 = datetime(2026, 6, 15, 14, 30, 0, tzinfo=UTC)
    bb.on_quote("SPY", t0, bid=99.99, bid_size=300, ask=100.01, ask_size=100)
    assert bb.on_trade("SPY", t0, 100.01, 100) is None      # at ask -> buy
    assert bb.on_trade("SPY", t0.replace(second=20), 99.99, 50) is None   # at bid -> sell
    assert bb.on_trade("SPY", t0.replace(second=40), 100.05, 25) is None  # uptick -> buy
    done = bb.flush("SPY", t0 + timedelta(minutes=1))
    assert done is not None
    assert (done.open, done.high, done.low, done.close) == (100.01, 100.05, 99.99, 100.05)
    assert done.volume == 175 and done.trade_count == 3
    expected_vwap = (100.01 * 100 + 99.99 * 50 + 100.05 * 25) / 175
    assert abs(done.vwap - expected_vwap) < 1e-9
    assert abs(done.mean_spread - 0.02) < 1e-9
    assert abs(done.mean_quote_imbalance - 0.5) < 1e-9   # (300-100)/400
    assert abs(done.flow_imbalance - (125 - 50) / 175) < 1e-9


def test_bar_builder_no_partial_bars() -> None:
    bb = BarBuilder(interval_s=60)
    t0 = datetime(2026, 6, 15, 14, 30, 0, tzinfo=UTC)
    assert bb.on_trade("SPY", t0, 100.0, 10) is None
    assert bb.on_trade("SPY", t0.replace(second=59), 101.0, 10) is None
    done = bb.on_trade("SPY", t0 + timedelta(minutes=1), 102.0, 10)
    assert done is not None and done.close == 101.0  # bar 1 excludes the new tick


def test_aggregate_5m() -> None:
    bb = BarBuilder(interval_s=60)
    t0 = datetime(2026, 6, 15, 14, 30, 0, tzinfo=UTC)
    bars = []
    for i in range(5):
        ts = t0 + timedelta(minutes=i)
        bb.on_trade("SPY", ts, 100.0 + i, 10)
        b = bb.flush("SPY", ts + timedelta(minutes=1))
        if b:
            bars.append(b)
    assert len(bars) == 5
    agg = aggregate(bars, 300)
    assert agg is not None
    assert agg.open == 100.0 and agg.close == 104.0 and agg.volume == 50
    assert agg.ts == bars[0].ts and agg.interval_s == 300


@pytest.mark.parametrize("side,qty", [("buy", 3), ("sell", 3)])
async def test_round_trip_recorded(state, repo, mock_broker, side: str, qty: int) -> None:  # type: ignore[no-untyped-def]
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    om.on_fill("AAPL", side, qty, 100.0, "signal", {"momentum": 0.5})
    close_side = "sell" if side == "buy" else "buy"
    om.on_fill("AAPL", close_side, qty, 110.0, "signal")
    with repo.session() as s:
        from lobplatform.persistence.models import Trade

        trades = s.query(Trade).all()
    assert len(trades) == 1
    expected = (110.0 - 100.0) * qty * (1 if side == "buy" else -1)
    assert abs(trades[0].pnl - expected) < 1e-9
    assert "AAPL" not in state.positions
