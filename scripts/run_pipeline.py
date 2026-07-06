"""End-to-end pipeline: simulate -> features -> labels -> train -> backtest.

Usage:
    python scripts/run_pipeline.py --model tcn
    python scripts/run_pipeline.py --model deeplob
    python scripts/run_pipeline.py --model both --events 1000000

Artifacts land in runs/<model>_<timestamp>/:
    report.json    classification + backtest metrics
    history.json   training curve
    model.pt       best checkpoint
    plots.png      mid-price, training curve, confusion matrix, PnL
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lob.backtest import BacktestConfig, run_backtest, signal_from_labels
from lob.data import clean_snapshots, load_lobster, load_synthetic
from lob.evaluate import classification_report, format_report
from lob.features import make_features
from lob.labels import LabelConfig, class_distribution, make_labels, suggest_alpha
from lob.train import TrainConfig, predict, temporal_split, train_model


def run_one(model_name: str, args, sim, runs_dir: Path) -> dict:
    print(f"\n=== {model_name.upper()} ===")
    mode = "raw40" if model_name == "deeplob" else "extended"
    features, names = make_features(sim, mode=mode)
    mid = sim.mid
    n = len(features)

    splits = temporal_split(n, args.window, args.horizon)

    # Calibrate alpha on the training slice only, then label everything
    alpha = suggest_alpha(mid[: splits.train[-1] + 1], args.horizon,
                          target_flat=args.target_flat)
    labels = make_labels(mid, LabelConfig(horizon=args.horizon, alpha=alpha))
    dist = class_distribution(labels[splits.train])
    print(f"alpha={alpha:.2e}  train class dist: "
          f"down {dist['down']:.2%} / flat {dist['flat']:.2%} / up {dist['up']:.2%}")

    cfg = TrainConfig(model=model_name, window=args.window,
                      max_epochs=args.epochs, batch_size=args.batch_size)
    t0 = time.time()
    result = train_model(features, labels, cfg, splits=splits, horizon=args.horizon)
    train_secs = time.time() - t0

    # ---- test evaluation -------------------------------------------------- #
    x_norm = result.normalizer.transform(features)
    test_pos = splits.test[labels[splits.test] != -1]
    probs = predict(result.model, x_norm, test_pos, args.window)
    y_true = labels[test_pos]
    y_pred = probs.argmax(axis=1)
    report = classification_report(y_true, y_pred)
    print(format_report(report))

    # ---- backtest on the test segment ------------------------------------- #
    bid = sim.snapshots[test_pos, 2] * sim.tick_size
    ask = sim.snapshots[test_pos, 0] * sim.tick_size
    bt_cfg = BacktestConfig(prob_threshold=args.prob_threshold)
    bt = run_backtest(probs, bid, ask,
                      sim.buy_flow[test_pos], sim.sell_flow[test_pos], bt_cfg)
    bt_nofilter = run_backtest(
        probs, bid, ask, sim.buy_flow[test_pos], sim.sell_flow[test_pos],
        BacktestConfig(prob_threshold=args.prob_threshold, use_toxicity_filter=False),
    )
    oracle = run_backtest(
        signal_from_labels(labels[test_pos]), bid, ask,
        sim.buy_flow[test_pos], sim.sell_flow[test_pos], bt_cfg,
    )
    print("backtest (net):", json.dumps(bt.summary, indent=None))
    print(f"oracle net PnL (upper bound): {oracle.summary['final_net_pnl']:.2f}")

    # ---- artifacts ---------------------------------------------------------#
    out = runs_dir / f"{args.data}_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), out / "model.pt")
    full = {
        "model": model_name,
        "data": args.data,
        "feature_mode": mode,
        "alpha": alpha,
        "train_class_dist": dist,
        "train_seconds": round(train_secs, 1),
        "best_val_macro_f1": round(result.best_val_f1, 4),
        "test": report,
        "backtest_filtered": bt.summary,
        "backtest_unfiltered": bt_nofilter.summary,
        "backtest_oracle": oracle.summary,
    }
    (out / "report.json").write_text(json.dumps(full, indent=2))
    (out / "history.json").write_text(json.dumps(result.history, indent=2))
    _plot(out, sim, result.history, report, bt, bt_nofilter, oracle, model_name)
    print(f"artifacts -> {out}")
    return full


def _plot(out, sim, history, report, bt, bt_nofilter, oracle, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    mid = sim.mid * sim.tick_size
    ax.plot(mid, lw=0.4)
    ax.set_title("Simulated mid-price (full session)")
    ax.set_xlabel("snapshot")

    ax = axes[0, 1]
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history],
            label="train loss")
    ax2 = ax.twinx()
    ax2.plot([h["epoch"] for h in history], [h["val_macro_f1"] for h in history],
             color="tab:orange", label="val macro-F1")
    ax.set_title(f"{model_name}: training curve")
    ax.set_xlabel("epoch")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1, 0]
    cm = np.array(report["confusion_matrix"], dtype=np.float64)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center",
                    color="black" if cm_norm[i, j] < 0.6 else "white")
    ax.set_xticks([0, 1, 2], ["down", "flat", "up"])
    ax.set_yticks([0, 1, 2], ["down", "flat", "up"])
    ax.set_title("Confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    ax.plot(bt.net_pnl, label=f"net, toxicity filter (Sharpe {bt.sharpe:.1f})")
    ax.plot(bt_nofilter.net_pnl, label="net, no filter", alpha=0.7)
    ax.plot(bt.gross_pnl, label="gross", alpha=0.5, ls="--")
    ax.plot(oracle.net_pnl, label="oracle net (upper bound)", alpha=0.5, ls=":")
    ax.set_title("Backtest PnL (test segment)")
    ax.set_xlabel("snapshot")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "plots.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tcn", choices=["tcn", "deeplob", "both"])
    p.add_argument("--data", default="synthetic", choices=["synthetic", "lobster"])
    p.add_argument("--orderbook", help="LOBSTER orderbook CSV (for --data lobster)")
    p.add_argument("--message", help="LOBSTER message CSV (for --data lobster)")
    p.add_argument("--events", type=int, default=1_000_000)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--target-flat", type=float, default=0.4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--prob-threshold", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    t0 = time.time()
    if args.data == "lobster":
        if not (args.orderbook and args.message):
            p.error("--data lobster requires --orderbook and --message")
        print(f"loading LOBSTER data ...")
        sim = load_lobster(args.orderbook, args.message)
        print(f"  {len(sim.timestamps):,} snapshots in {time.time()-t0:.1f}s, "
              f"session span {(sim.timestamps[-1]-sim.timestamps[0])/60:.1f} minutes")
    else:
        print(f"simulating {args.events:,} events ...")
        sim = load_synthetic(args.events, args.snapshot_every, seed=args.seed)
        print(f"  {len(sim.timestamps):,} snapshots in {time.time()-t0:.1f}s, "
              f"session span {sim.timestamps[-1]/60:.1f} sim-minutes")

    # Real-data hygiene step (near no-op on clean synthetic; essential for
    # LOBSTER, which carries crossed/locked/anomalous quotes).
    sim, clean_report = clean_snapshots(sim)
    print(f"  cleaning: {clean_report['n_bad']} bad snapshots "
          f"(crossed={clean_report['n_crossed']}, "
          f"wide={clean_report['n_wide_spread']})")

    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    models = ["tcn", "deeplob"] if args.model == "both" else [args.model]
    for m in models:
        run_one(m, args, sim, runs_dir)


if __name__ == "__main__":
    main()
