"use client";
// Closed-trade blotter with expandable entry-signal breakdown.
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

type Trade = {
  symbol: string; side: string; qty: number; entry_ts: string; exit_ts: string;
  entry_price: number; exit_price: number; pnl: number;
  holding_seconds: number; signal_scores: Record<string, unknown>;
};

export default function Trades() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [open, setOpen] = useState<number | null>(null);

  useEffect(() => {
    api<Trade[]>("/trades?limit=200").then(setTrades).catch(() => {});
  }, []);

  return (
    <main className="mx-auto max-w-5xl space-y-4 p-6">
      <h1 className="text-xl font-bold text-white">Trades</h1>
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-gray-500">
            <tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry → Exit</th><th>Held</th><th>P&L</th></tr>
          </thead>
          <tbody className="text-gray-200">
            {trades.map((t, i) => (
              <>
                <tr
                  key={i}
                  onClick={() => setOpen(open === i ? null : i)}
                  className="cursor-pointer border-t border-gray-800 hover:bg-gray-800/50"
                >
                  <td className="py-2 font-medium">{t.symbol}</td>
                  <td className="capitalize">{t.side}</td>
                  <td>{t.qty}</td>
                  <td>${t.entry_price.toFixed(2)} → ${t.exit_price.toFixed(2)}</td>
                  <td>{formatHold(t.holding_seconds)}</td>
                  <td className={t.pnl >= 0 ? "text-emerald-400" : "text-red-400"}>${t.pnl.toFixed(2)}</td>
                </tr>
                {open === i && (
                  <tr className="border-t border-gray-800 bg-black/40">
                    <td colSpan={6} className="px-4 py-3 text-xs text-gray-400">
                      <b>Entry signals:</b>{" "}
                      {Object.entries(t.signal_scores ?? {}).map(([k, v]) => (
                        <span key={k} className="mr-4">
                          {k}: {typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </span>
                      ))}
                      {Object.keys(t.signal_scores ?? {}).length === 0 && "none recorded"}
                    </td>
                  </tr>
                )}
              </>
            ))}
            {trades.length === 0 && (
              <tr><td colSpan={6} className="py-4 text-center text-gray-600">no closed trades yet</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}

function formatHold(s: number): string {
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}
