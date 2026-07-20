# LOB Platform — risk-tiered algorithmic paper-trading

A complete paper-trading platform grown out of an LOB (limit order book) ML
research project: a strategy engine executing against **Alpaca paper**, a
FastAPI control plane, and a Next.js dashboard.

**PAPER TRADING ONLY.** The broker layer refuses to start unless the base
URL is Alpaca's paper endpoint. No real money can be traded by this build.
No deposits, no other people's funds — ever (that is a regulated activity
and permanently out of scope).

```
┌─────────────┐   IEX ws    ┌──────────────────────────────────────┐
│ Alpaca      │────────────►│ ENGINE (lobctl)                      │
│ paper API   │◄────────────│ stream→bars→signals→ensemble         │
└─────────────┘   orders    │      ↓                               │
                            │ RiskManager.approve() ← EVERY order  │
                            │      ↓            KillSwitch (first) │
                            │ OrderManager → broker.py (paper-only │
                            │                assertion)            │
                            └──────┬───────────────────────────────┘
                                   │ shared DB (SQLite/Postgres) + commands table
                            ┌──────┴────────┐        ┌─────────────┐
                            │ API (FastAPI) │◄──────►│ WEB (Next)  │
                            │ REST + /ws    │  JWT   │ dashboard   │
                            └───────────────┘        └─────────────┘
```

## Quickstart

```bash
make setup                      # venv + deps
cp .env.example .env            # add your Alpaca PAPER keys + APP_PASSWORD
make test                       # 87 python tests (engine + api)
make run-engine                 # lobctl run --tier medium
make run-api                    # FastAPI on :8000 (docs at /docs)
make run-web                    # Next.js on :3000
```

Optional: `docker compose up -d` for Postgres+Redis (otherwise SQLite +
in-process fallbacks are automatic).

## Safety model

- **Kill switch built first**: daily-loss / drawdown / data-staleness /
  broker-error / manual triggers → persist HALTED → cancel all → flatten →
  verify flat → CRITICAL event. HALTED survives restarts; only an explicit
  `lobctl reset` or `POST /engine/reset` clears it.
- **No order without approval**: the broker is only reachable through the
  OrderManager, which requires an `Approval` token issued exclusively by
  `RiskManager.approve()` (or `approve_exit` for stops/EOD/kill flatten).
- **Risk tiers** (LOW/MEDIUM/HIGH) are frozen constants in
  `engine/lobplatform/config/tiers.py` — position caps, gross caps, loss
  limits, stops, cadence, signal weights.
- **Leakage discipline**: signals see only finalized bars; walk-forward
  training with day-granular embargo; normalizer stats from train only;
  model causality is unit-tested.

## The lob_flow signal

The centerpiece: the TCN (dilated-causal convolutions) from the original
research (`research/lob/models.py`), ported to
`engine/lobplatform/signals/nn/` with the same focal loss, class weighting,
and training discipline, retrained on 14 L1 microstructure features from
5-minute bars. `make train-signal` runs walk-forward training;
`make evaluate-signal` runs the §5.9 cost-aware quality gate
(experiments budget: 24 configs, immutable `models/experiments.csv`,
one-shot holdout). Live shadow evaluation halves a signal's ensemble
weight when its rolling 20-session performance degrades.

The original 10-level LOBSTER research remains intact in `research/`
(43 tests) as the offline research track.

## Repo layout

```
research/   original LOB ML project (untouched, still green)
engine/     lobplatform package: config, data, signals(+nn), risk,
            execution, persistence, engine loop, lobctl CLI
api/        FastAPI app + tests
web/        Next.js 14 dashboard (login, dashboard, trades, signals,
            settings) + Playwright smoke test
```

## Dashboard

Login (password → httpOnly JWT cookie) → dashboard: equity curve (live via
WebSocket), day P&L, positions with stops, engine status pill, tier
selector with confirm dialog, and a kill switch that requires typing
`FLATTEN`.

## Testing

- `make test` — 87 tests: sizing property tests, 100% branch coverage of
  risk rejections, kill-switch sequence, bar-builder correctness, NN port
  suite (shape/causality/overfit/focal/TorchScript/leakage), replay
  integration with a MockBroker (incl. stop/exit race regressions), full
  API surface + WS + metrics goldens.
- `make test-research` — 43 research-track tests (leakage, simulator,
  backtest math, xsec pipeline).
- `cd web && npx playwright test` — E2E smoke.
- CI: `.github/workflows/ci.yml` (ruff, mypy, pytest).
- Deployment: see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Going live (not built)

Live trading is Phase 4, deliberately ungated so far: this build hard-fails
outside paper mode. The design for owner-only live mode (separate keys,
triple opt-in, hard limits, paper-record gate, audit log) is specified in
the project plan but intentionally not implemented until Phases 1–3 are
accepted in real paper sessions. Managing other people's money requires
registration (RIA/broker-dealer); this project stays permanently on the
self-directed side of that line.

## License

Apache 2.0 — see [LICENSE](LICENSE).
