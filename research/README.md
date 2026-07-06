# LOB — Limit Order Book Mid-Price Prediction

A complete research pipeline for predicting short-horizon mid-price direction
from limit order book snapshots: simulator → features → labels → models
(DeepLOB, TCN) → leakage-safe training → cost-aware backtest with an
adverse-selection filter.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy pandas matplotlib scikit-learn pytest

# run the test suite
.venv/bin/python -m pytest tests/ -q

# full pipeline on synthetic data (both models)
.venv/bin/python scripts/run_pipeline.py --model both --events 1000000
```

Artifacts (metrics JSON, training history, checkpoint, plots) land in
`runs/<model>_<timestamp>/`.

## Project layout

```
lob/
  simulator.py   Event-driven LOB simulator: Cont-Stoikov-Talreja book with
                 Hawkes (self-exciting) market orders and slow regime drift,
                 stabilized by four mechanisms (see below)
  features.py    raw40 (DeepLOB layout) and 62-feature extended mode:
                 per-level imbalance, microprice, OFI, signed flow, RV
  labels.py      DeepLOB smooth labels (horizon k, threshold alpha),
                 alpha calibration on the train slice
  models.py      DeepLOB (conv blocks + inception + LSTM) and TCN
                 (dilated causal convs); both [B,T,F] -> 3-class logits
  losses.py      Focal loss with inverse-frequency class weights
  train.py       Temporal splits with embargo, windowed dataset,
                 training loop (early stopping on val macro-F1)
  evaluate.py    Per-class precision/recall/F1, confusion matrix
  backtest.py    Spread-crossing execution, fees + slippage, VPIN-style
                 toxicity filter, Sharpe / drawdown / hit-rate
  data.py        Adapters: synthetic, FI-2010, LOBSTER
scripts/
  run_pipeline.py  End-to-end CLI
tests/             49 assertions across simulator, features, labels,
                   models, splits, and backtest accounting
```

## Simulator stability (hard-won lessons)

A naive zero-intelligence book is bistable: depending on flow balance it
either thickens until the mid freezes, or hollows out and produces
flash-crash teleports. Four mechanisms keep this one in a realistic regime,
and each was added because its absence produced a concrete artifact:

1. **Price-band execution** (`price_band_ticks`): a marketable order only
   fills within N ticks of the pre-trade best, like real exchange price
   protection (LULD). Level-count caps do NOT work — levels can be
   arbitrarily far apart, which is exactly when you need the cap.
2. **Subcritical Hawkes** (`hawkes_alpha/kappa` well below branching ratio
   1, plus a hard cap on excitation): near-critical self-excitation
   produces heavy-tailed bursts that march the book hundreds of ticks.
3. **Spread-responsive liquidity** (`spread_response`): limit order
   intensity scales with the spread — the queue-reactive restoring force.
   Wide spread → quoting floods in → spread closes.
4. **Fundamental anchor** (`anchor_beta`): a slow random-walk value process
   tilts market buy/sell intensity against displacement. Bounds the speed
   of directional excursions so the book can thicken behind a move;
   without it, reversals teleport through hollow price ranges.

## Methodology notes (the parts most public implementations get wrong)

**Leakage discipline.** Splits are contiguous in time (train < val < test)
with an embargo gap of `window + horizon` samples between segments, so no
input window or label window straddles a boundary. The feature normalizer
and the label threshold `alpha` are both fitted on the training slice only.
The last `horizon` samples of the series are marked INVALID rather than
silently labeled.

**Label construction.** Smooth labels per the DeepLOB paper: compare the
mean of the next `k` mids against the current mid, threshold at `alpha`.
`suggest_alpha` calibrates the threshold to a target flat-class fraction
*on training data*; changing `k`/`alpha` changes the difficulty of the
problem, so report them with any accuracy number.

**Class imbalance.** Focal loss (γ=2) with inverse-frequency class weights.
Plain cross-entropy on imbalanced LOB labels produces an all-flat model with
deceptively high accuracy.

**Cost-aware evaluation.** The backtest crosses the spread on every entry
and exit, charges fees + slippage in bps, and reports gross vs. net PnL.
An oracle run (perfect-foresight labels through the same execution) gives
the strategy's upper bound — if your model's net PnL is a meaningful
fraction of oracle's, the signal is real; if gross is positive but net is
not, costs ate the edge (the usual outcome at short horizons).

**Adverse selection.** A VPIN-style toxicity measure (|net signed flow| /
total flow over a trailing window) suppresses new entries when flow is
one-sided — those are the moments an informed trader is running over the
book and a marketable order is most likely to be on the wrong side.

## Using real data

The whole pipeline is source-agnostic past `lob/data.py`; every adapter
returns the same `SimResult` interface.

**FI-2010** (free, academic benchmark): download from
[the FI-2010 repository](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)
and use `load_fi2010(path)`. Caveats: the public files are already z-scored
(skip normalizer fitting), there is no trade stream (toxicity filter
disabled), and it is one Helsinki market from 2010 — results do not
transfer to modern US equities.

**LOBSTER** (NASDAQ ITCH-derived, free samples at
[lobsterdata.com](https://lobsterdata.com)): use
`load_lobster(orderbook_csv, message_csv)`. This gives real 10-level books
plus executions, so the full pipeline — including the toxicity filter —
applies.

**Synthetic** (default): the simulator produces order-flow clustering
(Hawkes), regime drift, and realistic book mechanics, which is enough to
validate the pipeline end-to-end. It is still a zero-intelligence model:
expect real-data accuracy to be lower and cost sensitivity to be higher.

## Honest limitations

- Synthetic data has no real alpha decay, queue position effects, or
  venue fragmentation; treat results as pipeline validation, not strategy
  validation.
- The backtest assumes unit size and full fills at the touch — fine for
  signal evaluation, not for capacity analysis.
- Latency is not modeled: in production the signal must clear the spread
  *after* the time it takes you to act, which is strictly harder.
