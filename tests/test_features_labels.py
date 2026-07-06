import numpy as np
import pytest

from lob.features import Normalizer, build_raw40, make_features, split_book
from lob.labels import DOWN, FLAT, INVALID, UP, LabelConfig, make_labels, suggest_alpha
from lob.simulator import LOBSimulator, SimConfig


@pytest.fixture(scope="module")
def sim():
    return LOBSimulator(SimConfig(seed=3)).run(20_000, 5)


def test_raw40_price_centering(sim):
    x = build_raw40(sim.snapshots)
    # ask offsets positive, bid offsets negative
    assert np.all(x[:, 0::4] > 0)
    assert np.all(x[:, 2::4] < 0)
    # best ask offset + best bid offset = 0 by construction of mid
    np.testing.assert_allclose(x[:, 0] + x[:, 2], 0.0, atol=1e-9)


def test_extended_features(sim):
    x, names = make_features(sim, mode="extended")
    assert x.shape == (len(sim.timestamps), 62)
    assert len(names) == 62
    assert not np.any(np.isnan(x))
    assert not np.any(np.isinf(x))
    # imbalance features bounded in [-1, 1]
    imb_cols = [i for i, n in enumerate(names) if n.startswith("imb_l")]
    assert np.all(np.abs(x[:, imb_cols]) <= 1.0 + 1e-9)


def test_normalizer_train_only():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(1000, 4))
    nz = Normalizer().fit(x[:600])
    out = nz.transform(x[:600])
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-5)


def test_labels_known_series():
    # mid ramps up then flat then down: labels must follow
    mid = np.concatenate([
        np.linspace(100.0, 101.0, 200),   # rising
        np.full(200, 101.0),              # flat
        np.linspace(101.0, 100.0, 200),   # falling
    ])
    cfg = LabelConfig(horizon=10, alpha=1e-5)
    lab = make_labels(mid, cfg)
    assert lab[50] == UP
    assert lab[300] == FLAT
    assert lab[450] == DOWN
    # last k samples invalid
    assert np.all(lab[-10:] == INVALID)
    assert np.all(lab[:-10] != INVALID)


def test_suggest_alpha_balances(sim):
    mid = sim.mid
    alpha = suggest_alpha(mid, horizon=20, target_flat=0.4)
    lab = make_labels(mid, LabelConfig(horizon=20, alpha=alpha))
    valid = lab[lab != INVALID]
    flat_frac = (valid == FLAT).mean()
    assert 0.3 < flat_frac < 0.5, f"flat fraction {flat_frac:.3f}"


def test_split_book_roundtrip(sim):
    ap, av, bp, bv = split_book(sim.snapshots)
    assert ap.shape == (len(sim.timestamps), 10)
    np.testing.assert_array_equal(ap[:, 0], sim.snapshots[:, 0])
    np.testing.assert_array_equal(bv[:, 9], sim.snapshots[:, 39])
