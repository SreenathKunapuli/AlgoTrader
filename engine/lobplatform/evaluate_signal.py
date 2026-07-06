"""`make evaluate-signal` — the §5.9 cost-aware walk-forward harness.

Full-stack evaluation: signal -> ensemble-style gating -> MEDIUM-tier
sizing -> simulated fills at bar close ± half spread. Data is split
train / validation (walk-forward) with a FINAL HOLDOUT of the most recent
10 trading days written to `data_holdout/` — a directory the training path
never reads; it is evaluated exactly once per winner (§5.9 rule 3).

Quality gate: directional accuracy > 51.5% on non-flat predictions AND
net expectancy per trade > 0 after costs at MEDIUM settings.
Every candidate config appends one row to models/experiments.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .data.bar_builder import Bar
from .signals.nn.features import build_features
from .signals.nn.labels import DOWN, INVALID, UP, calibrate_alpha, make_labels
from .signals.nn.splits import walk_forward_day_split
from .signals.nn.training import SEQ_LEN, train

log = structlog.get_logger()

GATE_DIR_ACC = 0.515
FEE_SLIP_BPS = 0.8  # fees + slippage per side, bps
MAX_EXPERIMENTS = 24


@dataclass
class EvalMetrics:
    directional_accuracy: float
    macro_f1: float
    expectancy_per_trade: float
    sharpe: float
    max_drawdown: float
    turnover: float
    n_trades: int
    passes_gate: bool

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def simulate_trades(preds: np.ndarray, labels: np.ndarray, closes: np.ndarray,
                    spreads: np.ndarray, horizon: int = 3) -> EvalMetrics:
    """Fill at close ± half spread; hold `horizon` bars; MEDIUM-tier costs."""
    pnls: list[float] = []
    equity = [0.0]
    correct = total = 0
    for i in range(len(preds) - horizon):
        if labels[i] == INVALID or preds[i] == 1:
            continue
        direction = 1 if preds[i] == UP else -1
        if labels[i] != 1:
            total += 1
            if (preds[i] == UP and labels[i] == UP) or (preds[i] == DOWN and labels[i] == DOWN):
                correct += 1
        entry = closes[i] + direction * spreads[i] / 2       # cross half spread
        exit_ = closes[i + horizon] - direction * spreads[i + horizon] / 2
        fees = (entry + exit_) * FEE_SLIP_BPS * 1e-4
        pnl = direction * (exit_ - entry) - fees
        pnls.append(pnl)
        equity.append(equity[-1] + pnl)
    eq = np.array(equity)
    rets = np.diff(eq)
    sharpe = (float(rets.mean() / rets.std() * np.sqrt(252 * 78))
              if len(rets) > 2 and rets.std() > 0 else 0.0)
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) > 1 else 0.0
    exp_ = float(np.mean(pnls)) if pnls else 0.0
    dir_acc = correct / total if total else 0.0
    # macro-f1 over the 3 classes on valid samples
    valid = labels != INVALID
    yp, yt = preds[valid], labels[valid]
    f1s = []
    for c in range(3):
        tp = int(np.sum((yp == c) & (yt == c)))
        fp = int(np.sum((yp == c) & (yt != c)))
        fn = int(np.sum((yp != c) & (yt == c)))
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0)
    return EvalMetrics(
        directional_accuracy=dir_acc, macro_f1=float(np.mean(f1s)),
        expectancy_per_trade=exp_, sharpe=sharpe, max_drawdown=dd,
        turnover=float(len(pnls)), n_trades=len(pnls),
        passes_gate=(dir_acc > GATE_DIR_ACC and exp_ > 0),
    )


def log_experiment(models_dir: str, config: dict[str, Any], metrics: EvalMetrics) -> int:
    """Append one immutable row to experiments.csv; returns total rows."""
    path = Path(models_dir) / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["ts", "config_hash", "config_json", *metrics.as_dict().keys()])
        w.writerow([datetime.now(UTC).isoformat(), cfg_hash,
                    json.dumps(config, sort_keys=True), *metrics.as_dict().values()])
    with open(path) as f:
        n = sum(1 for _ in f) - 1
    if n > MAX_EXPERIMENTS:
        log.warning("experiments.budget_exceeded", n=n, budget=MAX_EXPERIMENTS)
    return n


def evaluate(bars_by_symbol: dict[str, list[Bar]], models_dir: str = "models",
             arch: str = "tcn", horizon: int = 3,
             holdout_dir: str = "data_holdout") -> EvalMetrics:
    """Train walk-forward and evaluate on validation. Writes experiments.csv.

    The most recent 10 distinct days per symbol are EXCLUDED from all
    train/val arrays and dumped to `holdout_dir` for the one-shot final
    check (run manually via evaluate_holdout after model selection).
    """
    hd = Path(holdout_dir)
    hd.mkdir(exist_ok=True)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    closes_l: list[np.ndarray] = []
    spreads_l: list[np.ndarray] = []
    tr_idx: list[int] = []
    va_idx: list[int] = []
    base = 0
    for sym, bars in bars_by_symbol.items():
        days = sorted({b.ts.date() for b in bars})
        if len(days) < 16:
            continue
        holdout_days = set(days[-10:])
        work = [b for b in bars if b.ts.date() not in holdout_days]
        hold = [b for b in bars if b.ts.date() in holdout_days]
        (hd / f"{sym}.json").write_text(json.dumps(
            [{"ts": b.ts.isoformat(), "o": b.open, "h": b.high, "l": b.low,
              "c": b.close, "v": b.volume, "vw": b.vwap, "tc": b.trade_count,
              "sp": b.mean_spread, "qi": b.mean_quote_imbalance,
              "fi": b.flow_imbalance} for b in hold]))
        x = build_features(work)
        closes = np.array([b.close for b in work])
        stamps = [b.ts for b in work]
        sp = walk_forward_day_split(stamps)
        alpha = calibrate_alpha(closes[sp.train_idx], horizon)
        y = make_labels(closes, horizon, alpha)
        xs.append(x)
        ys.append(y)
        closes_l.append(closes)
        spreads_l.append(np.array([b.mean_spread for b in work]))
        tr_idx += [i + base for i in sp.train_idx]
        va_idx += [i + base for i in sp.val_idx]
        base += len(x)
    if not xs:
        raise ValueError("no symbols with >=16 days of data")
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    result = train(x_all, y_all, tr_idx, va_idx, arch=arch)

    # validation predictions -> cost-aware simulation (pooled)
    import torch

    model = result.model.eval()
    va = sorted(va_idx)
    va = [i for i in va if i >= SEQ_LEN - 1 and y_all[i] != INVALID]
    windows = np.stack([x_all[i - SEQ_LEN + 1: i + 1] for i in va])
    with torch.no_grad():
        preds = model(torch.from_numpy(windows)).argmax(1).numpy()
    close_flat = np.concatenate(closes_l)
    spread_flat = np.concatenate(spreads_l)
    metrics = simulate_trades(preds, y_all[va], close_flat[va], spread_flat[va], horizon)
    config = {"arch": arch, "horizon": horizon, "seq_len": SEQ_LEN,
              "gate": {"dir_acc": GATE_DIR_ACC}}
    n = log_experiment(models_dir, config, metrics)
    report = Path(models_dir) / "lob_flow_report.md"
    report.write_text(
        f"# lob_flow evaluation\n\nconfigs tried: {n}/{MAX_EXPERIMENTS}\n\n"
        f"## Validation (walk-forward)\n```json\n{json.dumps(metrics.as_dict(), indent=2)}\n```\n\n"
        f"Gate ({'PASS' if metrics.passes_gate else 'FAIL'}): dir_acc > {GATE_DIR_ACC} "
        f"and expectancy > 0.\n\n"
        + ("" if metrics.passes_gate else
           "## Signal is weak\nThe candidate failed the quality gate; per §5.9 it ships at "
           "reduced ensemble weight (confidence scaling in lob_flow.py). With "
           f"{n} configs tried, treat marginal wins as likely luck.\n")
        + "\nHoldout: evaluate exactly once after model selection "
          "(data in data_holdout/, never read by training).\n")
    return metrics
