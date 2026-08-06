# LOB Platform

An algorithmic trading platform that started as a limit-order-book prediction
research project and grew into a full engine-to-dashboard system. It has been
running live against an **Alpaca paper trading account since July 2026** —
no real money involved, and this README makes no claim about profitability.
The point of the live run is to prove the engine, risk controls, and
broker-reconciliation logic hold up outside a backtest, not to demonstrate a
winning strategy.

The repo has two halves. `research/` is the original offline ML pipeline:
a Hawkes-process order book simulator, a 62-feature extraction layer, and
TCN/DeepLOB models trained on it and on 1.4M real LOBSTER order-book
snapshots, evaluated with a cost-aware backtester. `engine/` + `api/` +
`web/` is what came after: an asyncio trading engine that turns one of
those models (plus two simpler signals) into an ensemble, a FastAPI control
plane, and a Next.js dashboard to watch and control it.

## Architecture at a glance

```
Alpaca paper API (IEX websocket + REST)
        │  bars, quotes, trades
        ▼
engine/lobplatform/          asyncio trading engine (lobctl CLI)
  data/alpaca_stream.py        websocket ingestion, budgeted subscriptions
  data/bar_builder.py          tick -> 5-min bar aggregation
  signals/lob_flow.py          TCN signal (ported from research/lob/models.py)
  signals/momentum.py,
  signals/mean_reversion.py    the other two ensemble members
  signals/ensemble.py          Kaufman-ER regime blending of the three
  strategy/xsec_momentum.py    monthly cross-sectional momentum book
  risk/risk_manager.py         sole path to order approval
  risk/kill_switch.py          daily-loss / drawdown / staleness / manual halt
  execution/order_manager.py   cancel-replace, stop tracking, EOD flatten
  execution/broker.py          Alpaca client — refuses to start on a live URL
  execution/reconcile.py       broker-state reconciliation on wake/restart
  persistence/                 SQLite models + repo (commands table for control)
        │  shared DB
        ▼
api/app/                     FastAPI control plane
  main.py                       REST endpoints + /ws/stream (DB-poll fallback)
  auth.py                       JWT, httpOnly cookie
        │  REST + WebSocket
        ▼
web/                          Next.js 14 dashboard
  app/{dashboard,trades,signals,positions,settings}
  hooks/useStream.ts            WS hook driving the live equity chart
  e2e/smoke.spec.ts             Playwright: login -> dashboard -> kill dialog
```

Config is not a database table: `engine/lobplatform/config/tiers.py` defines
LOW/MEDIUM/HIGH risk tiers (position caps, gross caps, loss limits, stop
distances) as frozen dataclasses, so changing a risk limit is a reviewable
diff, not a runtime toggle. No order reaches the broker without an
`Approval` token issued by `RiskManager.approve()`.

## How to run it

```bash
make setup                      # venv + deps (engine, api, research)
cp .env.example .env            # Alpaca PAPER keys + APP_PASSWORD

make test                       # 87 tests: engine + api
make test-research              # 43 tests: research track

make run-engine                 # lobctl run --tier medium
make run-api                    # FastAPI on :8000 (docs at /docs)
make run-web                    # Next.js on :3000
```

`make test` and `make test-research` are separate invocations because the
research pipeline (`research/`) is otherwise unrelated code with its own
`requirements.txt` — I verified both by running them: 87 passed / 43 passed,
no skips.

Optional: `docker compose up -d` brings up Postgres + Redis; without it, the
engine falls back to SQLite and in-process pubsub automatically.

## Hard problems

**Three live sessions, zero orders.** After deploying the ML signal live, the
engine ran for three sessions — 820 signal computes in the database by the
end (the fix commit quotes 720, the tally when diagnosis began) — without
placing a single order. Three separate bugs stacked: the
`directional_accuracy` metric
was scored recall-style, so a FLAT prediction on a moving bar counted as a
failure and had quietly zeroed the signal's confidence at export time; the
momentum and mean-reversion signals disagreed on direction in about 45% of
the computes where both were active (only ~5% of all computes) and
cancelled each other out in the ensemble (fixed by switching to
Kaufman efficiency-ratio regime blending instead of a fixed average); and
the tier gate thresholds assumed score levels the ensemble could never
actually reach. Fixing the metric alone wouldn't have been enough — all
three had to go (`db49f28`).

**A stop-loss double-execution on live paper money.** On 2026-07-16 a
cancel-replace on an exit order double-executed: the code cancelled the old
order and immediately computed a replacement quantity, but Alpaca's cancel
is asynchronous, so the "cancelled" order was still live when the
replacement went out, and both filled. The fix makes cancel-replace poll the
order to a terminal state before sizing the replacement, adds a one-exit-
in-flight-per-symbol guard (the stop checker and the EOD flattener were
racing each other), and makes the resulting position — not the order side —
determine which way a new stop should point, since a partial scale-down sell
on a long had been restaging a short-style stop above the current price
(`20bd894`).

**A causality test that could never fail.** The TCN's whole reason for
existing is that it can't see the future — that's what makes it usable
live instead of just in a backtest. The unit test for that property fed the
model two inputs it had made byte-identical before the forward pass even
ran (it sliced off the perturbed future *before* calling the model), so the
assertion could never trigger regardless of whether the property held. It
had been passing the entire time it wasn't testing anything. The rewritten
test perturbs the future on a full-length input, checks the perturbation
actually reaches later internal features (so it can't go vacuous silently
again), and only then asserts the earlier prediction is unchanged
(`39d4bd0`).

## Notes on the research track

`research/` predates the live platform and stands on its own — a Cont-
Stoikov-Talreja order book simulator with Hawkes self-exciting order flow,
kept in a stable (subcritical) regime by four mechanisms documented in
`research/README.md`. One of them, `price_band_ticks`, exists because an
earlier version capped how many book levels a marketable order could walk
through — which doesn't work, because levels can be arbitrarily far apart
in price, which is exactly when a price-protection band is needed. The
`lob_flow` signal running live today is the TCN from this track, retrained
on 14 microstructure features instead of the original 62. See
`research/README.md` for the full pipeline (simulator, features, labels,
models, leakage-safe splits, cost-aware backtest with an oracle upper
bound).
