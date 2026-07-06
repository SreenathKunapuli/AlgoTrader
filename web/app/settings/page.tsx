"use client";
// Read-only view of active tier parameters (env-driven items shown, not editable).
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const TIERS: Record<string, Record<string, string>> = {
  low: {
    Universe: "10 ETFs (SPY, QQQ, IWM, DIA, XLK, XLF, XLE, XLV, GLD, TLT)",
    Direction: "Long only", "Max position": "5% of equity", "Max gross": "40%",
    "Max open positions": "6", "Daily loss kill": "1.0%", "Max drawdown kill": "5%",
    Stop: "1.5 × ATR(14, 5m)", "Confidence threshold": "0.60",
    Rebalance: "daily at 15:45 ET", "Risk per trade": "0.25%",
  },
  medium: {
    Universe: "10 ETFs + 10 large caps", Direction: "Long only",
    "Max position": "10% of equity", "Max gross": "80%", "Max open positions": "10",
    "Daily loss kill": "2.0%", "Max drawdown kill": "10%", Stop: "2.0 × ATR",
    "Confidence threshold": "0.55", Rebalance: "every 15 min", "Risk per trade": "0.50%",
  },
  high: {
    Universe: "10 ETFs + 10 large caps", Direction: "Long + short",
    "Max position": "20% of equity", "Max gross": "150%", "Max open positions": "15",
    "Daily loss kill": "4.0%", "Max drawdown kill": "15%", Stop: "2.5 × ATR",
    "Confidence threshold": "0.52", Rebalance: "every 5 min", "Risk per trade": "1.00%",
  },
};

export default function Settings() {
  const [tier, setTier] = useState<string>("medium");
  const [mode, setMode] = useState<string>("paper");

  useEffect(() => {
    api<{ tier: string; trading_mode: string }>("/engine/status")
      .then((s) => { setTier(s.tier); setMode(s.trading_mode); })
      .catch(() => {});
  }, []);

  return (
    <main className="mx-auto max-w-3xl space-y-4 p-6">
      <h1 className="text-xl font-bold text-white">Settings</h1>
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4 text-sm">
        <div className="mb-3 text-gray-400">
          Trading mode: <b className="text-sky-300 uppercase">{mode}</b> · Active tier:{" "}
          <b className="capitalize text-emerald-400">{tier}</b>
        </div>
        <table className="w-full">
          <tbody>
            {Object.entries(TIERS[tier] ?? {}).map(([k, v]) => (
              <tr key={k} className="border-t border-gray-800">
                <td className="py-2 pr-4 text-gray-500">{k}</td>
                <td className="text-gray-200">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-4 text-xs text-gray-600">
          Tier parameters are code-reviewed constants (engine/lobplatform/config/tiers.py) —
          not editable from the UI by design. Secrets and endpoints live in .env.
        </p>
      </div>
    </main>
  );
}
