"""Regression tests for the 2026-07-09 'engine never trades' fixes:
prediction-conditioned directional accuracy (dead lob_flow), regime-blended
momentum/mean-reversion weights, and passive (maker-only) entry execution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from lobplatform.config.tiers import TIERS, Tier
from lobplatform.data.bar_builder import Bar
from lobplatform.execution.broker import BrokerOrder
from lobplatform.execution.order_manager import OrderManager
from lobplatform.risk.risk_manager import OrderIntent, RiskManager
from lobplatform.signals.ensemble import Ensemble, trend_weight
from lobplatform.signals.mean_reversion import MeanReversionSignal
from lobplatform.signals.momentum import MomentumSignal
from lobplatform.signals.nn.training import directional_accuracy

MED = TIERS[Tier.MEDIUM]
IN_SESSION = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)


# ---------- directional accuracy: abstentions are not failures ---------- #
def test_directional_accuracy_prediction_conditioned() -> None:
    # model calls 2 directions on moving bars, both right; abstains (FLAT)
    # on two other moving bars; one call lands on a flat bar (excluded)
    yt = np.array([2, 0, 2, 0, 1])
    yp = np.array([2, 0, 1, 1, 2])
    assert directional_accuracy(yt, yp) == 1.0
    # wrong-direction calls count against it
    yp_bad = np.array([0, 2, 1, 1, 2])
    assert directional_accuracy(yt, yp_bad) == 0.0
    # all-flat predictions: no evidence, 0.0 (not a crash)
    assert directional_accuracy(yt, np.ones(5, dtype=int)) == 0.0


def test_incumbent_metrics_revive_lob_flow() -> None:
    """The recall-style metric read 0.3276 and zeroed lob_flow's confidence;
    the corrected metric on the same confusion matrix clears the 0.51 bar."""
    cm = [[943, 988, 549], [1042, 3403, 751], [892, 1031, 743]]
    yt = np.concatenate([np.full(sum(cm[i]), i) for i in range(3)])
    yp = np.concatenate([np.repeat([0, 1, 2], cm[i]) for i in range(3)])
    da = directional_accuracy(yt, yp)
    assert abs(da - 0.5392) < 5e-4
    assert da > 0.51  # lob_flow ships at full confidence


# ---------- regime blending ---------- #
def _bar(ts: datetime, close: float) -> Bar:
    return Bar(symbol="SPY", ts=ts, interval_s=300, open=close, high=close * 1.0002,
               low=close * 0.9998, close=close, volume=1000, vwap=close,
               trade_count=50, mean_spread=0.02, mean_quote_imbalance=0.0,
               flow_imbalance=0.0)


def _bars(closes: list[float]) -> list[Bar]:
    t0 = IN_SESSION - timedelta(minutes=5 * len(closes))
    return [_bar(t0 + timedelta(minutes=5 * i), c) for i, c in enumerate(closes)]


def test_trend_weight_extremes() -> None:
    trending = _bars([100.0 + 0.1 * i for i in range(30)])       # ER = 1
    choppy = _bars([100.0 + 0.1 * (i % 2) for i in range(30)])   # ER ~ 0
    assert trend_weight(trending) == 1.0
    assert trend_weight(choppy) == 0.0
    assert trend_weight(_bars([100.0] * 5)) == 0.5  # insufficient data: neutral


def test_ensemble_reallocates_momentum_mr_pool() -> None:
    ens = Ensemble([MomentumSignal(), MeanReversionSignal()])
    pool = MED.signal_weights["momentum"] + MED.signal_weights["mean_reversion"]

    res_trend = ens.compute("SPY", _bars([100.0 + 0.1 * i for i in range(40)]), MED)
    assert res_trend.per_signal["momentum"]["weight"] == pool
    assert res_trend.per_signal["mean_reversion"]["weight"] == 0.0

    res_chop = ens.compute("SPY", _bars([100.0 + 0.1 * (i % 2) for i in range(40)]), MED)
    assert res_chop.per_signal["mean_reversion"]["weight"] == pool
    assert res_chop.per_signal["momentum"]["weight"] == 0.0

    # pool is conserved either way
    for res in (res_trend, res_chop):
        total = (res.per_signal["momentum"]["weight"]
                 + res.per_signal["mean_reversion"]["weight"])
        assert abs(total - pool) < 1e-12


# ---------- passive entries vs urgent exits ---------- #
async def test_entry_posts_at_near_touch(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    rm = RiskManager(MED, state)
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    ap = rm.approve(OrderIntent(symbol="AAPL", side="buy", qty=5, price_hint=100.0),
                    IN_SESSION)
    assert not ap.urgent  # type: ignore[union-attr]
    order = await om.submit(ap, IN_SESSION, mid=100.0, spread=0.10)  # type: ignore[arg-type]
    assert order is not None and order.side == "buy"
    submitted = mock_broker.submitted[-1]
    with repo.session() as s:
        from lobplatform.persistence.models import Order

        row = s.query(Order).filter_by(broker_order_id=submitted.id).one()
    assert abs(row.limit_price - 99.95) < 1e-9  # mid - spread/2: earn, don't pay


async def test_exit_crosses_toward_touch(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    from lobplatform.risk.state import Position

    state.positions["AAPL"] = Position(symbol="AAPL", qty=5, entry_price=100.0, mark=100.0)
    rm = RiskManager(MED, state)
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    ap = rm.approve_exit(OrderIntent(symbol="AAPL", side="sell", qty=5,
                                     price_hint=100.0, reason="stop"))
    assert ap.urgent  # type: ignore[union-attr]
    await om.submit(ap, IN_SESSION, mid=100.0, spread=0.10)  # type: ignore[arg-type]
    with repo.session() as s:
        from lobplatform.persistence.models import Order

        row = s.query(Order).filter_by(symbol="AAPL").one()
    assert abs(row.limit_price - 99.99) < 1e-9  # mid - 10% of spread


async def test_unfilled_entry_repegs_then_expires_never_market(  # type: ignore[no-untyped-def]
        state, repo, mock_broker) -> None:
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    stale = BrokerOrder(id="o-stale", client_order_id="c" * 32, symbol="AAPL",
                        side="buy", qty=5, status="new", filled_qty=0)
    mock_broker.open_orders.append(stale)

    await om._repeg_entry(stale, "signal", mid=100.0)
    assert "o-stale" in mock_broker.cancelled
    repegged = mock_broker.submitted[-1]
    assert repegged.client_order_id.endswith("-rp")
    last_row_types = [o for o in mock_broker.submitted]
    assert all(o.client_order_id != "market" for o in last_row_types)

    # second leg unfilled -> expire: cancel only, still no market order
    mock_broker.open_orders.append(BrokerOrder(
        id=repegged.id, client_order_id=repegged.client_order_id, symbol="AAPL",
        side="buy", qty=5, status="new", filled_qty=0))
    n_submitted = len(mock_broker.submitted)
    await om._expire_entry(repegged)
    assert repegged.id in mock_broker.cancelled
    assert len(mock_broker.submitted) == n_submitted  # nothing new submitted


async def test_unfilled_exit_still_goes_market(state, repo, mock_broker) -> None:  # type: ignore[no-untyped-def]
    om = OrderManager(mock_broker, repo, state)  # type: ignore[arg-type]
    stale = BrokerOrder(id="o-exit", client_order_id="d" * 32, symbol="AAPL",
                        side="sell", qty=5, status="new", filled_qty=0)
    mock_broker.open_orders.append(stale)
    await om._replace_if_unfilled(stale, "stop")
    assert "o-exit" in mock_broker.cancelled
    with repo.session() as s:
        from lobplatform.persistence.models import Order

        types = [o.order_type for o in s.query(Order).all()]
    assert "market" in types
