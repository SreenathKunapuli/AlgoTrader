"""Full-universe walk-forward tournament: momentum vs GBT vs MLP vs ensemble
vs distilled student, on ~3000 liquid US stocks with point-in-time top-2000
eligibility. The scoreboard is one cost-aware portfolio and rank IC.

Usage: python -m research.scripts.run_xsec2  (after fetch_universe.py)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from research.xsec import backtest as bt                    # noqa: E402
from research.xsec import features as ft                    # noqa: E402
from research.xsec.data import fetch_daily, to_panels       # noqa: E402
from research.xsec.models import (DistilledStudent, GBTRanker, MLPRanker,  # noqa: E402
                                  TeacherEnsemble)

OOS_START_YEAR = 2020
ELIGIBLE_TOP = 2000
TOP_N = 100  # of ~2000 eligible = top 5%


def main() -> None:
    syms = pd.read_csv(ROOT / "data/universe3000.csv")["symbol"].tolist() + ["SPY"]
    raw = fetch_daily(syms, str(ROOT / "data/xsec_daily_3000.pkl"))
    panels = to_panels(raw)
    close, volume = panels["close"], panels["volume"]
    spy = close.pop("SPY")
    volume = volume.drop(columns=["SPY"], errors="ignore")
    print(f"panel: {close.shape[0]} days x {close.shape[1]} symbols, "
          f"{close.index[0].date()} .. {close.index[-1].date()}", flush=True)

    dates = ft.month_end_dates(close.index)
    ds = ft.build_dataset(close, volume, dates, market=spy, eligible_top=ELIGIBLE_TOP)
    feats = [c for c in ft.ALL_FEATURES if c in ds.columns]
    print(f"dataset: {len(ds):,} samples, {ds['date'].nunique()} dates, "
          f"{len(feats)} features", flush=True)

    oos_years = sorted({int(d.year) for d in dates if d.year >= OOS_START_YEAR})

    contenders: dict[str, dict] = {}
    gbt_models = bt.yearly_models(ds, oos_years, GBTRanker, feats)
    print("gbt trained", flush=True)
    mlp_models = bt.yearly_models(ds, oos_years, MLPRanker, feats)
    print("mlp trained", flush=True)
    teachers = {y: TeacherEnsemble() for y in oos_years}
    train = ds.dropna(subset=["fwd_ret", "y_rank"])
    students: dict[int, DistilledStudent] = {}
    for y in oos_years:
        sub = train[train["date"] <= bt.train_cutoff(y)]
        x = sub[feats].values
        teachers[y].gbt, teachers[y].mlp = gbt_models[y], mlp_models[y]  # reuse fits
        students[y] = DistilledStudent(teachers[y]).fit(x)
    print("teachers assembled, students distilled", flush=True)

    contenders["momentum"] = {"models": gbt_models, "momentum_only": True}
    contenders["gbt"] = {"models": gbt_models}
    contenders["mlp"] = {"models": mlp_models}
    contenders["teacher_ens"] = {"models": teachers}
    contenders["distilled"] = {"models": students}

    results: dict[str, pd.Series] = {}
    summaries = []
    spy_ret_full = spy.pct_change()
    for name, cfg in contenders.items():
        scores = bt.predict_scores(ds, cfg["models"],
                                   momentum_only=cfg.get("momentum_only", False),
                                   features=feats)
        ic = bt.information_coefficient(scores)
        res = bt.run_portfolio(scores, close, top_n=TOP_N)
        results[name] = res.daily_returns
        spy_ret = spy_ret_full.reindex(res.daily_returns.index).fillna(0.0)
        s = bt.summarize(res.daily_returns, name, benchmark=spy_ret)
        s.update(ic)
        s["turnover"] = float(res.turnover.mean())
        summaries.append(s)
        print(f"{name}: done", flush=True)

    idx = results["momentum"].index
    spy_ret = spy_ret_full.reindex(idx).fillna(0.0)
    ew_ret = close.pct_change().mean(axis=1).reindex(idx).fillna(0.0)
    results["spy"] = spy_ret
    results["ew_universe"] = ew_ret
    summaries.append(bt.summarize(spy_ret, "spy"))
    summaries.append(bt.summarize(ew_ret, "ew_universe", benchmark=spy_ret))

    best = max((s for s in summaries if s["label"] in contenders),
               key=lambda s: s["sharpe"])["label"]
    vm = bt.vol_managed(results[best])
    results[f"{best}_volmgd"] = vm
    summaries.append(bt.summarize(vm, f"{best}_volmgd", benchmark=spy_ret))

    print("\n=== tournament (OOS, net of 10bps one-way) ===")
    cols = ["label", "cagr", "vol", "sharpe", "max_dd", "ann_active", "ir",
            "ic_mean", "ic_tstat", "ic_hit", "turnover"]
    table = pd.DataFrame(summaries).set_index("label").reindex(columns=[c for c in cols if c != "label"])
    print(table.round(4).to_string())

    print("\n=== per-year returns ===")
    per = bt.per_year_table(results).round(4)
    print(per.to_string())

    print("\n=== distilled student coefficients (last year) ===")
    print(students[oos_years[-1]].coefficients(feats).round(4).to_string())

    outdir = ROOT / "data/xsec_results"
    outdir.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(outdir / "tournament_daily_returns.csv")
    table.to_csv(outdir / "tournament_summary.csv")
    per.to_csv(outdir / "tournament_per_year.csv")
    print(f"\nwrote {outdir}/")


if __name__ == "__main__":
    main()
