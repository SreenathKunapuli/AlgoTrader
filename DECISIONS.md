# Decisions log

- [2026-07-05] Use existing `.venv` (Python 3.13) — satisfies the 3.11+ requirement; torch already installed, avoids a second heavy install. `uv`/`pnpm` unavailable → pip + npm per decision table fallback.
- [2026-07-05] Moved `data/` and `runs/` under `research/` — they are research artifacts; scripts use CWD-relative paths and run from `research/` unchanged. CSVs are gitignored (450 MB).
- [2026-07-05] Single root `pyproject.toml` governs engine+api (one venv, one tool config) instead of per-package projects — simplest monorepo layout that keeps `research/` untouched.
- [2026-07-05] Feature count F=14 fixed as: quote_imb, flow_imb, rel_spread, logret_1, logret_3, logret_6, logret_12, vwap_dev, vol_z, range_atr, tradecount_z, rv_pctile, tod_sin, tod_cos (spec said "~14").
- [2026-07-05] Approval-token enforcement: OrderManager.submit(approval: Approval); Approval only constructible via RiskManager._issue — structural §1.6 guarantee.
- [2026-07-05] LOW-tier daily rebalance approximated as rebalance_seconds=86400 within the entry window rather than a 15:45 ET scheduler — simplest correct behavior; revisit if LOW is used heavily.
- [2026-07-06] Web: browser gets the WS token from a cookie-gated same-origin /api/ws-token route (httpOnly cookie can't be read by JS; the backend WS requires a query token). Single-user tradeoff.
- [2026-07-06] Playwright smoke mocks the backend at the network layer (page.route) so it runs standalone; cookie is seeded explicitly because route.fulfill set-cookie doesn't populate the browser jar. Real auth path covered by API tests.
- [2026-07-06] lightweight-charts v5 API (addSeries(LineSeries, ...)) used instead of the v4 addLineSeries in older docs.
- [2026-07-06] Alpaca Basic plan caps ws subscriptions at 30/connection (hit in first live run: 20 symbols x trades+quotes = 40 -> 405 -> staleness kill, safety worked). Live path now subscribes the official 1-min BAR channel for all symbols and spends leftover budget on quotes then trades for leading symbols; local BarBuilder remains for tests/replay. Symbols without quote/trade subs carry zero spread/imbalance/flow features.\n