"use client";
// Latest ensemble per symbol with per-signal contribution bars.
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStream } from "@/hooks/useStream";

type Sig = { symbol: string; ts: string; ensemble: number; per_signal: Record<string, unknown> };

export default function Signals() {
  const [rows, setRows] = useState<Sig[]>([]);
  const { latest } = useStream(["signals"]);

  useEffect(() => {
    api<Sig[]>("/signals/latest").then(setRows).catch(() => {});
  }, []);

  useEffect(() => {
    const s = latest.signals as Sig | undefined;
    if (s?.symbol) {
      setRows((prev) => {
        const rest = prev.filter((r) => r.symbol !== s.symbol);
        return [{ symbol: s.symbol, ts: s.ts, ensemble: s.ensemble, per_signal: s.per_signal }, ...rest];
      });
    }
  }, [latest]);

  return (
    <main className="mx-auto max-w-5xl space-y-4 p-6">
      <h1 className="text-xl font-bold text-white">Signals</h1>
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-gray-500">
            <tr><th>Symbol</th><th>Ensemble</th><th>Breakdown</th><th>As of</th></tr>
          </thead>
          <tbody className="text-gray-200">
            {rows.sort((a, b) => Math.abs(b.ensemble) - Math.abs(a.ensemble)).map((r) => (
              <tr key={r.symbol} className="border-t border-gray-800">
                <td className="py-2 font-medium">{r.symbol}</td>
                <td className={r.ensemble >= 0 ? "text-emerald-400" : "text-red-400"}>
                  {r.ensemble.toFixed(3)}
                </td>
                <td className="py-2">
                  <div className="flex gap-3">
                    {Object.entries(r.per_signal ?? {}).map(([name, v]) => {
                      const rec = v as { contribution?: number } | number;
                      const contrib = typeof rec === "object" ? (rec.contribution ?? 0) : Number(rec);
                      const w = Math.min(Math.abs(contrib) * 100, 50);
                      return (
                        <div key={name} className="flex items-center gap-1 text-xs text-gray-400">
                          {name}
                          <div className="h-2 w-14 overflow-hidden rounded bg-gray-800">
                            <div
                              className={`h-full ${contrib >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
                              style={{ width: `${w * 2}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </td>
                <td className="text-xs text-gray-500">{new Date(r.ts).toLocaleTimeString()}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td colSpan={4} className="py-4 text-center text-gray-600">no signals yet</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}
