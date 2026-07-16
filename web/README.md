# LOB web dashboard

Next.js dashboard for the trading engine: live equity chart, open positions,
signal feed, trade history, and engine controls (tier selector, kill switch).
Data arrives over a WebSocket stream from the FastAPI backend; REST fills in
the initial page state.

## Run

```bash
npm install
npm run dev   # http://localhost:3000
```

The FastAPI backend must be running (default `http://localhost:8000`).
Override with a `web/.env.local`:

```
NEXT_PUBLIC_API_BASE=http://localhost:8000
NEXT_PUBLIC_WS_BASE=ws://localhost:8000
API_INTERNAL_BASE=http://localhost:8000
```

## Pages

| Route | Contents |
|---|---|
| `/` | Equity + day P&L cards, live equity chart, positions table, tier + kill controls |
| `/positions` | Open positions with entry signals, stops, and age |
| `/signals` | Live ensemble signal feed per symbol |
| `/trades` | Closed-trade history with P&L |
| `/settings` | Engine configuration view |
| `/login` | Password login (JWT session cookie) |

## Tests

```bash
npx playwright test   # smoke tests (dev server + API must be running)
```
