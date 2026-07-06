"""Inference latency benchmark.

For live trading the budget is end-to-end: feature construction for the
newest snapshot + model forward pass. At a 1-second prediction horizon the
whole loop should clear 50ms with comfortable margin; at sub-second
horizons, 10ms. This measures both pieces for batch size 1 (the live case).

Usage:
    python scripts/benchmark_inference.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lob.data import load_synthetic
from lob.features import make_features
from lob.models import build_model
from lob.train import pick_device


def bench_forward(model, x, device, n_iter=300, warmup=50) -> np.ndarray:
    model.eval()
    times = np.empty(n_iter)
    with torch.no_grad():
        for i in range(warmup + n_iter):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times[i - warmup] = time.perf_counter() - t0
    return times * 1e3   # ms


def bench_features(sim, n_iter=100) -> float:
    """Incremental cost approximated by full-vector cost / N (amortized);
    plus we measure a single 200-row tail rebuild, the realistic live shape."""
    tail = sim.snapshots[-200:]
    bf, sf = sim.buy_flow[-200:], sim.sell_flow[-200:]
    from lob.features import build_extended

    t0 = time.perf_counter()
    for _ in range(n_iter):
        build_extended(tail, bf, sf)
    return (time.perf_counter() - t0) / n_iter * 1e3


def main():
    device = pick_device()
    print(f"device: {device}")

    sim = load_synthetic(50_000, 5)
    feat_ms = bench_features(sim)
    print(f"feature rebuild (200-row tail, 62 features): {feat_ms:.3f} ms")

    for name, f in (("deeplob", 40), ("tcn", 62)):
        model = build_model(name, f).to(device)
        x = torch.randn(1, 100, f, device=device)
        times = bench_forward(model, x, device)
        p50, p99 = np.percentile(times, [50, 99])
        total50 = p50 + feat_ms
        verdict = "OK <50ms" if total50 < 50 else "OVER 50ms budget"
        print(f"{name:8s}  forward p50 {p50:6.2f} ms  p99 {p99:6.2f} ms"
              f"  | end-to-end p50 ~{total50:5.2f} ms  [{verdict}]")

    # CPU comparison matters: for batch-1 inference, GPU dispatch overhead
    # often makes CPU faster
    if device.type != "cpu":
        cpu = torch.device("cpu")
        for name, f in (("deeplob", 40), ("tcn", 62)):
            model = build_model(name, f).to(cpu)
            x = torch.randn(1, 100, f)
            times = bench_forward(model, x, cpu)
            p50 = np.percentile(times, 50)
            print(f"{name:8s}  forward p50 {p50:6.2f} ms  (cpu)")


if __name__ == "__main__":
    main()
