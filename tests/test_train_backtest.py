import numpy as np
import pytest

from lob.backtest import BacktestConfig, run_backtest, signal_from_labels, vpin_toxicity
from lob.labels import INVALID, LabelConfig, make_labels
from lob.train import SplitIndices, WindowDataset, temporal_split


def test_temporal_split_embargo():
    s = temporal_split(10_000, window=100, horizon=20)
    embargo = 120
    assert s.train[-1] + embargo < s.val[0] + 1
    assert s.val[-1] + embargo < s.test[0] + 1
    # train starts only after a full input window exists
    assert s.train[0] == 99
    # test labels never need future data beyond the series
    assert s.test[-1] + 20 < 10_000


def test_window_dataset_drops_invalid():
    x = np.random.randn(500, 8).astype(np.float32)
    y = np.ones(500, dtype=np.int64)
    y[200:210] = INVALID
    pos = np.arange(100, 400)
    ds = WindowDataset(x, y, pos, window=50)
    assert len(ds) == 300 - 10
    win, lab = ds[0]
    assert win.shape == (50, 8)
    assert lab == 1


def test_window_dataset_causal():
    """The window for position t must end exactly at t."""
    x = np.arange(100, dtype=np.float32)[:, None]
    y = np.zeros(100, dtype=np.int64)
    ds = WindowDataset(x, y, np.array([60]), window=10)
    win, _ = ds[0]
    np.testing.assert_array_equal(win[:, 0].numpy(), np.arange(51, 61))


def test_vpin_bounds():
    rng = np.random.default_rng(0)
    b = rng.exponential(100, 1000)
    s = rng.exponential(100, 1000)
    tox = vpin_toxicity(b, s, 50)
    assert np.all(tox >= 0) and np.all(tox <= 1 + 1e-9)
    # one-sided flow must read as maximally toxic
    tox_onesided = vpin_toxicity(b, np.zeros(1000), 50)
    np.testing.assert_allclose(tox_onesided, 1.0)


def test_oracle_backtest_profitable():
    """Perfect-foresight signals must be profitable gross of costs.

    If the oracle loses money gross, the execution accounting is broken.
    """
    rng = np.random.default_rng(1)
    n = 5000
    mid = 10_000 + np.cumsum(rng.standard_normal(n) * 2.0)
    spread = 1.0
    bid, ask = mid - spread / 2, mid + spread / 2

    labels = make_labels(mid, LabelConfig(horizon=20, alpha=1e-5))
    probs = signal_from_labels(labels)
    flows = rng.exponential(50, n)
    res = run_backtest(
        probs, bid, ask, flows, flows,
        BacktestConfig(use_toxicity_filter=False, prob_threshold=0.55),
    )
    assert res.gross_pnl[-1] > 0, f"oracle gross PnL {res.gross_pnl[-1]:.1f}"
    assert res.n_trades > 10
    assert res.total_costs > 0


def test_flat_signal_no_trades():
    n = 1000
    probs = np.full((n, 3), [0.1, 0.8, 0.1])
    bid = np.full(n, 99.0)
    ask = np.full(n, 101.0)
    res = run_backtest(probs, bid, ask, np.ones(n), np.ones(n))
    assert res.n_trades == 0
    assert res.net_pnl[-1] == 0.0


def test_toxicity_filter_suppresses():
    rng = np.random.default_rng(2)
    n = 2000
    mid = 10_000 + np.cumsum(rng.standard_normal(n))
    bid, ask = mid - 0.5, mid + 0.5
    probs = np.tile([0.05, 0.05, 0.90], (n, 1))     # always wants to be long
    buy = np.full(n, 100.0)                          # perfectly one-sided flow
    sell = np.zeros(n)
    res_filt = run_backtest(probs, bid, ask, buy, sell,
                            BacktestConfig(use_toxicity_filter=True))
    res_nofilt = run_backtest(probs, bid, ask, buy, sell,
                              BacktestConfig(use_toxicity_filter=False))
    assert res_filt.suppressed_entries > 0
    assert res_filt.n_trades < res_nofilt.n_trades
