import numpy as np
import pytest

from lob.simulator import LOBSimulator, SimConfig


@pytest.fixture(scope="module")
def result():
    cfg = SimConfig(seed=42)
    return LOBSimulator(cfg).run(n_events=20_000, snapshot_every=5)


def test_shapes(result):
    n = 20_000 // 5
    assert result.snapshots.shape == (n, 40)
    assert result.timestamps.shape == (n,)
    assert result.buy_flow.shape == (n,)
    assert result.sell_flow.shape == (n,)


def test_book_integrity(result):
    s = result.snapshots
    ask_p = s[:, 0::4]
    ask_v = s[:, 1::4]
    bid_p = s[:, 2::4]
    bid_v = s[:, 3::4]

    # no crossed or locked book
    assert np.all(ask_p[:, 0] > bid_p[:, 0])
    # ask prices strictly increasing across levels, bids strictly decreasing
    assert np.all(np.diff(ask_p, axis=1) > 0)
    assert np.all(np.diff(bid_p, axis=1) < 0)
    # volumes strictly positive at every reported level
    assert np.all(ask_v > 0)
    assert np.all(bid_v > 0)


def test_time_and_flow(result):
    assert np.all(np.diff(result.timestamps) > 0)
    assert np.all(result.buy_flow >= 0)
    assert np.all(result.sell_flow >= 0)
    # Hawkes market orders must actually fire on both sides
    assert result.buy_flow.sum() > 0
    assert result.sell_flow.sum() > 0


def test_mid_price_moves(result):
    mid = result.mid
    # price must actually move (not a frozen book) ...
    assert np.std(mid) > 0.5
    # ... but not explode: stays within 20% of the seed mid
    assert np.all(np.abs(mid - 10_000) < 2_000)


def test_no_teleporting_mid(result):
    """No flash-crash artifacts: a single-snapshot mid move is bounded by
    roughly snapshot_every * price_band_ticks / 2 plus spread slack."""
    jumps = np.abs(np.diff(result.mid))
    assert jumps.max() <= 30, f"max 1-step mid jump {jumps.max():.1f} ticks"


def test_reproducible():
    a = LOBSimulator(SimConfig(seed=1)).run(5_000, 5).snapshots
    b = LOBSimulator(SimConfig(seed=1)).run(5_000, 5).snapshots
    np.testing.assert_array_equal(a, b)


def test_flow_clustering(result):
    """Hawkes excitation should make signed flow autocorrelated."""
    flow = result.buy_flow - result.sell_flow
    flow = flow - flow.mean()
    ac1 = np.corrcoef(flow[:-1], flow[1:])[0, 1]
    assert ac1 > 0.01, f"expected positive flow autocorrelation, got {ac1:.4f}"
