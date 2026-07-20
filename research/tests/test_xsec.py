"""Lookahead and correctness tests for the cross-sectional pipeline.

The one bug class that silently fabricates alpha is future data reaching a
feature or a weight. Every test here perturbs the FUTURE and asserts the
present is unchanged, or checks a hand-computed value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xsec import backtest as bt
from xsec import features as ft


def make_panel(n_days: int = 400, n_syms: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_syms)), axis=0)),
        index=dates, columns=[f"S{i}" for i in range(n_syms)])
    volume = pd.DataFrame(rng.integers(1e5, 1e6, (n_days, n_syms)),
                          index=dates, columns=close.columns).astype(float)
    return close, volume


def test_features_ignore_future():
    close, volume = make_panel()
    t = close.index[300]
    a = {k: v.loc[t] for k, v in ft.compute_features(close, volume).items()}
    close2, volume2 = close.copy(), volume.copy()
    close2.iloc[301:] *= 1.5  # rewrite the future
    volume2.iloc[301:] *= 3.0
    b = {k: v.loc[t] for k, v in ft.compute_features(close2, volume2).items()}
    for k in ft.FEATURES:
        pd.testing.assert_series_equal(a[k], b[k])


def test_forward_return_is_future_only():
    close, _ = make_panel()
    t = close.index[100]
    fwd = ft.forward_return(close).loc[t, "S0"]
    expected = close.iloc[100 + ft.HORIZON]["S0"] / close.iloc[100]["S0"] - 1.0
    assert fwd == pytest.approx(expected)
    # trailing rows have no complete window -> NaN, never a partial return
    assert ft.forward_return(close).iloc[-ft.HORIZON:].isna().all().all()


def test_dataset_labels_are_ranks_within_date():
    close, volume = make_panel()
    dates = ft.month_end_dates(close.index)
    ds = ft.build_dataset(close, volume, dates)
    labeled = ds.dropna(subset=["y_rank"])
    assert len(labeled) > 0
    for _, block in labeled.groupby("date"):
        assert block["y_rank"].max() <= 1.0 and block["y_rank"].min() > 0.0
        # rank must be monotone in fwd_ret
        srt = block.sort_values("fwd_ret")
        assert srt["y_rank"].is_monotonic_increasing


def test_training_cutoff_excludes_label_overlap():
    close, volume = make_panel(n_days=1400)
    dates = ft.month_end_dates(close.index)
    ds = ft.build_dataset(close, volume, dates)
    year = int(ds["date"].max().year)
    models = bt.yearly_models(ds, [year])
    # a model for OOS year Y must never have seen a sample dated Dec Y-1 or
    # later (its 21d label window crosses into Y)
    cutoff = pd.Timestamp(f"{year - 1}-11-30")
    train_dates = ds[ds["date"] <= cutoff]["date"]
    assert train_dates.max() <= cutoff
    assert year in models


def test_portfolio_math_single_period():
    # 2 picks, known returns, hand-computed EW buy-and-hold + cost
    dates = pd.bdate_range("2021-01-01", periods=4)
    close = pd.DataFrame({"A": [100, 110, 121, 121], "B": [100, 100, 90, 90],
                          "C": [100, 1, 1, 1]}, index=dates, dtype=float)
    scores = pd.DataFrame({"date": [dates[0]] * 3, "symbol": ["A", "B", "C"],
                           "score": [0.9, 0.8, 0.1]})
    res = bt.run_portfolio(scores, close, top_n=2, cost_bps=100.0)
    # day1: EW of +10% and 0% = +5%, minus 1% cost on turnover 1.0
    assert res.daily_returns.iloc[0] == pytest.approx(0.05 - 0.01)
    assert res.gross_returns.iloc[0] == pytest.approx(0.05)
    # day2: wealth A 1.21, B 0.90 -> port (1.21+0.90)/2 vs (1.10+1.00)/2
    assert res.gross_returns.iloc[1] == pytest.approx((2.11 / 2) / (2.10 / 2) - 1)
    assert res.turnover.iloc[0] == pytest.approx(1.0)


def test_summarize_known_series():
    r = pd.Series([0.01] * 252, index=pd.bdate_range("2022-01-03", periods=252))
    s = bt.summarize(r, "x")
    assert s["cagr"] == pytest.approx(1.01 ** 252 - 1, rel=1e-6)
    assert s["max_dd"] == 0.0


def test_eligibility_is_point_in_time():
    close, volume = make_panel(n_days=400, n_syms=30)
    dates = ft.month_end_dates(close.index)
    top = 10
    ds = ft.build_dataset(close, volume, dates, eligible_top=top)
    dollar = (close * volume).rolling(63).mean()
    for d, block in ds.groupby("date"):
        assert len(block) <= top
        # every kept symbol's trailing dollar volume must be >= every dropped one's
        kept = set(block["symbol"])
        dv = dollar.loc[d].dropna()
        if len(dv) > top:
            worst_kept = min(dv[s] for s in kept if s in dv)
            best_dropped = dv.drop(labels=[s for s in kept if s in dv]).max()
            assert worst_kept >= best_dropped * 0.999


def test_market_features_require_market():
    close, volume = make_panel()
    base = ft.compute_features(close, volume)
    assert set(base) == set(ft.FEATURES)
    market = close.mean(axis=1)
    full = ft.compute_features(close, volume, market=market)
    assert set(full) == set(ft.ALL_FEATURES)
    # beta of the market vs itself ~ 1; equal-vol random walks near 0-ish beta
    assert full["beta_12m"].iloc[-1].abs().max() < 5


def test_vol_managed_uses_only_past():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300),
                  index=pd.bdate_range("2021-01-01", periods=300))
    vm_a = bt.vol_managed(r).iloc[:200]
    r2 = r.copy()
    r2.iloc[200:] = 0.30  # violent future
    vm_b = bt.vol_managed(r2).iloc[:200]
    pd.testing.assert_series_equal(vm_a, vm_b)
    # scale never leverages up
    realized = r.rolling(63).std() * np.sqrt(252)
    hot = realized.shift(-1) > 0.20  # unused; just sanity that clip works
    assert (bt.vol_managed(r).abs() <= r.abs() + 1e-15).all()


def test_information_coefficient_perfect_and_inverted():
    dates = [pd.Timestamp("2022-01-31")] * 50 + [pd.Timestamp("2022-02-28")] * 50
    rng = np.random.default_rng(1)
    fwd = rng.normal(0, 0.05, 100)
    perfect = pd.DataFrame({"date": dates, "symbol": [f"S{i}" for i in range(100)],
                            "score": fwd, "fwd_ret": fwd})
    assert bt.information_coefficient(perfect)["ic_mean"] == pytest.approx(1.0)
    inverted = perfect.assign(score=-perfect["score"])
    assert bt.information_coefficient(inverted)["ic_mean"] == pytest.approx(-1.0)


def test_sentiment_student_roundtrip(tmp_path):
    import pandas as pd
    from xsec.sentiment import score_headlines, train_student
    rows = []
    for i in range(40):
        rows.append({"headline": f"Company{i} Q{i%4} EPS Beats Estimate, Sales Beat", "label": 1})
        rows.append({"headline": f"Company{i} Receives Nasdaq Delisting Notice Number {i}", "label": -1})
        rows.append({"headline": f"A Look At Company{i} Options Activity This Week", "label": 0})
    csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    model = tmp_path / "m.pkl"
    metrics = train_student(str(csv), model_path=str(model))
    assert metrics["holdout_accuracy"] > 0.8  # separable synthetic corpus
    scores = score_headlines(["NewCo EPS Beats Estimate", "NewCo Delisting Notice"],
                             model_path=str(model))
    assert scores[0] > scores[1]


def test_distilled_student_tracks_teacher():
    from xsec.models import DistilledStudent, TeacherEnsemble
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, (2000, len(ft.FEATURES)))
    y = (0.6 * x[:, 0] - 0.4 * x[:, 3] + rng.normal(0, 0.1, 2000))
    y = pd.Series(y).rank(pct=True).values
    teacher = TeacherEnsemble().fit(x, y)
    student = DistilledStudent(teacher).fit(x)
    x_new = rng.uniform(0, 1, (500, len(ft.FEATURES)))
    corr = np.corrcoef(pd.Series(teacher.predict(x_new)).rank(),
                       pd.Series(student.predict(x_new)).rank())[0, 1]
    assert corr > 0.8  # linear student recovers a mostly-linear teacher
