"""Subscription budget planning + stream-bar enrichment tests."""

from __future__ import annotations

from datetime import UTC, datetime

from lobplatform.data.alpaca_stream import SUBSCRIPTION_LIMIT, plan_subscriptions


def test_plan_within_limit_full_fidelity() -> None:
    syms = [f"S{i}" for i in range(10)]
    bars, quotes, trades = plan_subscriptions(syms)
    assert bars == syms and quotes == syms and trades == syms  # 30 exactly


def test_plan_medium_universe_fits() -> None:
    syms = [f"S{i}" for i in range(20)]
    bars, quotes, trades = plan_subscriptions(syms)
    assert bars == syms                 # every symbol gets bars
    assert len(quotes) == 10 and trades == []
    assert len(bars) + len(quotes) + len(trades) <= SUBSCRIPTION_LIMIT


def test_plan_oversized_universe_truncates() -> None:
    syms = [f"S{i}" for i in range(40)]
    bars, quotes, trades = plan_subscriptions(syms)
    assert len(bars) == 30 and quotes == [] and trades == []


async def test_on_stream_bar_enriches_and_feeds(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    from .test_replay import make_engine

    engine, _ = make_engine(state, repo, mock_broker)
    ts = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)
    # accumulate microstructure, then deliver the official bar
    await engine.on_quote("SPY", ts, 99.99, 300, 100.01, 100)
    await engine.on_trade("SPY", ts, 100.01, 100)   # at ask -> +100
    await engine.on_trade("SPY", ts, 99.99, 50)     # at bid -> -50
    await engine.on_stream_bar("SPY", ts, 100.0, 100.1, 99.9, 100.05, 150, 100.02, 2)

    bars = list(engine.bars_1m["SPY"])
    assert len(bars) == 1
    b = bars[0]
    assert b.close == 100.05 and b.volume == 150
    assert abs(b.mean_spread - 0.02) < 1e-9
    assert abs(b.mean_quote_imbalance - 0.5) < 1e-9
    assert abs(b.flow_imbalance - (100 - 50) / 150) < 1e-9
    # accumulators drained
    await engine.on_stream_bar("SPY", ts, 100.0, 100.1, 99.9, 100.0, 10, 100.0, 1)
    assert list(engine.bars_1m["SPY"])[-1].flow_imbalance == 0.0
