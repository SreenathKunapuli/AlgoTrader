import dataclasses

import numpy as np

from lob.data import clean_snapshots
from lob.simulator import LOBSimulator, SimConfig


def _make_result(n=200):
    return LOBSimulator(SimConfig(seed=11)).run(n * 5, 5)


def test_clean_noop_on_good_data():
    res = _make_result()
    cleaned, report = clean_snapshots(res)
    assert report["n_bad"] == 0
    # untouched
    np.testing.assert_array_equal(cleaned.snapshots, res.snapshots)


def test_clean_forward_fills_crossed():
    res = _make_result()
    s = res.snapshots.copy()
    good_row = s[40].copy()
    # inject a crossed book at row 41: best bid above best ask
    s[41, 0] = s[41, 2] - 5      # ask1 below bid1
    res = dataclasses.replace(res, snapshots=s)

    cleaned, report = clean_snapshots(res)
    assert report["n_crossed"] >= 1
    assert report["n_bad"] >= 1
    # row 41 replaced by last good row (40)
    np.testing.assert_array_equal(cleaned.snapshots[41], good_row)
    # a valid (non-crossed) book everywhere after cleaning
    assert np.all(cleaned.snapshots[:, 0] > cleaned.snapshots[:, 2])


def test_clean_drops_when_requested():
    res = _make_result()
    s = res.snapshots.copy()
    s[10, 2] = s[10, 0] + 100    # crossed
    res = dataclasses.replace(res, snapshots=s)
    cleaned, report = clean_snapshots(res, drop=True)
    assert len(cleaned.snapshots) == len(s) - report["n_bad"]
    assert len(cleaned.timestamps) == len(cleaned.snapshots)
    assert np.all(cleaned.snapshots[:, 0] > cleaned.snapshots[:, 2])


def test_clean_wide_spread_flagged():
    res = _make_result()
    s = res.snapshots.copy()
    # blow the spread far past the median multiple
    s[55, 0] = s[55, 2] + 100_000
    res = dataclasses.replace(res, snapshots=s)
    cleaned, report = clean_snapshots(res, max_spread_mult=50.0)
    assert report["n_wide_spread"] >= 1
    new_spread = cleaned.snapshots[55, 0] - cleaned.snapshots[55, 2]
    assert new_spread < 100_000
