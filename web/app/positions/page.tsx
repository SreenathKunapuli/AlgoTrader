"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStream } from "@/hooks/useStream";

type Position = {
  symbol: string;
  side: "long" | "short";
  qty: number;
  book: string;
  entry: number;
  mark: number;
  market_value: number;
  upnl: number;
  stop: number | null;
  age_s: number | null;
  entry_signals: Record<string, number>;
};

export default function Positions() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const { latest } = useStream(["positions"]);

  useEffect(() => {
    api<Position[]>("/positions").then(setPositions).catch(() => {});
  }, []);

  useEffect(() => {
    const livePos = latest.positions as { positions: Position[] } | undefined;
    if (livePos) setPositions(livePos.positions ?? []);
  }, [latest]);

  const totalUpnl = positions.reduce((s, p) => s + p.upnl, 0);
  const totalGross = positions.reduce((s, p) => s + Math.abs(p.market_value), 0);
  const longs = positions.filter((p) => p.qty > 0).length;
  const shorts = positions.filter((p) => p.qty < 0).length;

  return (
    <main className="mx-auto max-w-6xl space-y-4 p-6">
      <h1 className="text-xl font-bold text-white">Open Positions</h1>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Card label="Open positions" value={String(positions.length)} />
        <Card label="Long / Short" value={`${longs} / ${shorts}`} />
        <Card
          label="Unrealized P&L"
          value={`$${totalUpnl.toFixed(2)}`}
          tone={totalUpnl >= 0 ? "pos" : "neg"}
        />
        <Card label="Gross exposure" value={`$${totalGross.toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-gray-500">
            <tr>
              <th className="pb-2">Symbol</th>
              <th>Side</th>
              <th>Qty</th>
              <th>Book</th>
              <th>Entry</th>
              <th>Mark</th>
              <th>Mkt Value</th>
              <th>Unreal P&L</th>
              <th>Stop</th>
              <th>Age</th>
            </tr>
          </thead>
          <tbody className="text-gray-200">
            {positions.map((p) => (
              <>
                <tr
                  key={p.symbol}
                  onClick={() => setExpanded(expanded === p.symbol ? null : p.symbol)}
                  className="cursor-pointer border-t border-gray-800 hover:bg-gray-800/50"
                >
                  <td className="py-2 font-medium">{p.symbol}</td>
                  <td>
                    <span className={`rounded px-1.5 py-0.5 text-xs font-semibold ${p.qty > 0 ? "bg-emerald-900 text-emerald-300" : "bg-red-900 text-red-300"}`}>
                      {p.side}
                    </span>
                  </td>
                  <td>{Math.abs(p.qty)}</td>
                  <td>
                    <span className="rounded bg-sky-900 px-1.5 py-0.5 text-xs font-semibold text-sky-300">
                      Day Trade
                    </span>
                  </td>
                  <td>${p.entry?.toFixed(2)}</td>
                  <td>${p.mark?.toFixed(2)}</td>
                  <td>${p.market_value?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                  <td className={p.upnl >= 0 ? "text-emerald-400" : "text-red-400"}>
                    ${p.upnl?.toFixed(2)}
                  </td>
                  <td>{p.stop ? `$${p.stop.toFixed(2)}` : "—"}</td>
                  <td className="text-gray-500">{p.age_s != null ? formatAge(p.age_s) : "—"}</td>
                </tr>
                {expanded === p.symbol && (
                  <tr className="border-t border-gray-800 bg-black/40">
                    <td colSpan={10} className="px-4 py-3 text-xs text-gray-400">
                      <span className="font-semibold text-gray-300">Entry signals: </span>
                      {Object.keys(p.entry_signals ?? {}).length > 0
                        ? Object.entries(p.entry_signals).map(([k, v]) => (
                            <span key={k} className="mr-4">
                              {k}: <span className={Number(v) >= 0 ? "text-emerald-400" : "text-red-400"}>{Number(v).toFixed(3)}</span>
                            </span>
                          ))
                        : <span className="text-gray-600">none recorded</span>}
                    </td>
                  </tr>
                )}
              </>
            ))}
            {positions.length === 0 && (
              <tr>
                <td colSpan={10} className="py-8 text-center text-gray-600">
                  flat — no open positions
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}

function Card({ label, value, tone }: { label: string; value: string; tone?: "pos" | "neg" }) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${tone === "pos" ? "text-emerald-400" : tone === "neg" ? "text-red-400" : "text-white"}`}>
        {value}
      </div>
    </div>
  );
}

function formatAge(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}
