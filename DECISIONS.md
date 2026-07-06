# Decisions log

- [2026-07-05] Use existing `.venv` (Python 3.13) — satisfies the 3.11+ requirement; torch already installed, avoids a second heavy install. `uv`/`pnpm` unavailable → pip + npm per decision table fallback.
- [2026-07-05] Moved `data/` and `runs/` under `research/` — they are research artifacts; scripts use CWD-relative paths and run from `research/` unchanged. CSVs are gitignored (450 MB).
- [2026-07-05] Single root `pyproject.toml` governs engine+api (one venv, one tool config) instead of per-package projects — simplest monorepo layout that keeps `research/` untouched.
- [2026-07-05] Feature count F=14 fixed as: quote_imb, flow_imb, rel_spread, logret_1, logret_3, logret_6, logret_12, vwap_dev, vol_z, range_atr, tradecount_z, rv_pctile, tod_sin, tod_cos (spec said "~14").
