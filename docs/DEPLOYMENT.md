# Deployment

Three processes make up a running deployment: the **engine** (must run on a
single machine — Alpaca allows one data websocket per account), the **API**,
and the **web** dashboard. The engine and API share a database; everything
else is optional.

## 0. One-time setup

```bash
make setup                 # python venv + editable install
cp .env.example .env       # then edit:
```

Required `.env` values:

| Key | Value |
|-----|-------|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | your **paper** keys from app.alpaca.markets |
| `APP_PASSWORD` | dashboard login password (not `change-me`) |
| `JWT_SECRET` | long random string: `openssl rand -hex 32` |

Everything else has safe defaults (SQLite DB, in-process pub-sub, medium tier).

## 1. Verify before starting (every deploy)

```bash
make test                  # 87 engine+api tests
make lint && make typecheck
cd web && npx tsc --noEmit && cd ..
```

All must be green. Optionally `make test-research` (43 tests, ~40s).

## 2. Simplest deployment (laptop / single machine, SQLite)

Three terminals:

```bash
make run-engine            # terminal 1 — caffeinate keeps macOS awake
make run-api               # terminal 2 — http://localhost:8000/docs
make run-web               # terminal 3 — http://localhost:3000
```

Log in at http://localhost:3000 with `APP_PASSWORD`. Confirm:

1. Engine status pill shows **RUNNING** (not PAUSED/HALTED).
2. `curl localhost:8000/healthz` → `{"db":"ok","engine_heartbeat_fresh":true}`.
3. After the first session open: signals appear on the Signals page.

## 3. Docker deployment (Postgres + Redis + API + web)

```bash
docker compose up -d --build      # postgres, redis, api :8000, web :3000
```

Then run the engine **on the host** against the same services — add to `.env`:

```
DATABASE_URL=postgresql+psycopg2://lob:lob@localhost:5432/lob
REDIS_URL=redis://localhost:6379/0
```

```bash
make run-engine
```

The engine is deliberately not containerized: it needs exactly one instance
(the `lobengine.pid` lock enforces this per machine) and benefits from
`caffeinate` on macOS.

## 4. Operations

| Action | Command |
|--------|---------|
| Status | `.venv/bin/lobctl status` |
| Kill switch (cancel + flatten + halt) | `.venv/bin/lobctl halt` or dashboard FLATTEN |
| Clear HALTED after review | `.venv/bin/lobctl reset` |
| Retrain the lob_flow signal | `make train-signal` then `make evaluate-signal` (redeploy only on PASS) |
| Logs | `logs/engine_run.log` (or stdout, JSON lines) |

## 5. Safety invariants (do not "fix" these)

- The broker layer refuses to start unless the base URL is Alpaca's **paper**
  endpoint. Live trading is Phase 4 and intentionally unimplemented.
- HALTED survives restarts; only an explicit reset clears it.
- Every order requires a `RiskManager` approval token; tier limits are frozen
  constants in `engine/lobplatform/config/tiers.py`.

## Pre-deploy checklist

- [ ] `.env` has real paper keys, a real `APP_PASSWORD`, a random `JWT_SECRET`
- [ ] `make test` green (87), `make lint` + `make typecheck` clean
- [ ] `lobctl status` shows no stale HALTED you don't understand
- [ ] `models/lob_flow.pt` present (or accept lob_flow contributing zero)
- [ ] Only one engine instance anywhere (one Alpaca data connection per account)
- [ ] Machine won't sleep (macOS: `make run-engine` uses caffeinate)
