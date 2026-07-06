"""Shadow-eval tracker + §5.9 simulation/gate tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from lobplatform.evaluate_signal import log_experiment, simulate_trades
from lobplatform.signals.health import SignalHealthTracker
from lobplatform.signals.nn.labels import DOWN, FLAT, UP


def _add_trades(repo, pnls: list[float], signal: str = "lob_flow") -> None:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    for i, pnl in enumerate(pnls):
        repo.add_trade(symbol="SPY", side="long", qty=1,
                       entry_ts=now - timedelta(days=1), exit_ts=now - timedelta(hours=i),
                       entry_price=100.0, exit_price=100.0 + pnl, pnl=pnl,
                       signal_scores_json={signal: {"contribution": 0.5}})


def test_health_halves_weight_on_losses(repo) -> None:  # type: ignore[no-untyped-def]
    tracker = SignalHealthTracker(repo, ["lob_flow", "momentum"])
    _add_trades(repo, [-1.0] * 10)
    mults = tracker.evaluate()
    assert mults["lob_flow"] == 0.5           # degraded -> halved
    assert mults["momentum"] == 1.0           # no evidence -> untouched
    tracker.restore("lob_flow")
    assert tracker.multipliers["lob_flow"] == 1.0


def test_health_keeps_weight_on_wins(repo) -> None:  # type: ignore[no-untyped-def]
    tracker = SignalHealthTracker(repo, ["lob_flow"])
    _add_trades(repo, [1.0] * 10)
    assert tracker.evaluate()["lob_flow"] == 1.0


def test_simulate_trades_gate() -> None:
    n = 500
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 0.05, n))
    spreads = np.full(n, 0.01)
    # construct labels from actual future moves
    fut = np.roll(closes, -3) - closes
    labels = np.where(fut > 0.05, UP, np.where(fut < -0.05, DOWN, FLAT))
    labels[-3:] = -1
    # oracle predictions = labels -> should pass the gate on a liquid tape
    m = simulate_trades(labels.copy(), labels, closes, spreads)
    assert m.directional_accuracy == 1.0
    assert m.expectancy_per_trade > 0
    assert m.passes_gate
    # inverted predictions -> fail
    inv = np.where(labels == UP, DOWN, np.where(labels == DOWN, UP, FLAT))
    m2 = simulate_trades(inv, labels, closes, spreads)
    assert not m2.passes_gate


def test_experiments_csv_append(tmp_path) -> None:  # type: ignore[no-untyped-def]
    m = simulate_trades(np.array([1, 1, 1, -1]), np.array([1, 1, 1, -1]),
                        np.array([100.0, 100.1, 100.2, 100.3]), np.full(4, 0.01))
    n1 = log_experiment(str(tmp_path), {"arch": "tcn"}, m)
    n2 = log_experiment(str(tmp_path), {"arch": "tcn", "horizon": 6}, m)
    assert (n1, n2) == (1, 2)
    rows = (tmp_path / "experiments.csv").read_text().strip().splitlines()
    assert len(rows) == 3  # header + 2 immutable rows
