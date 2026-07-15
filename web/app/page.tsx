"use client";
// Dashboard: header cards, live equity chart, positions table, tier + kill.
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { useStream } from "@/hooks/useStream";
import EquityChart from "@/components/EquityChart";
import { KillButton, TierSelector } from "@/components/Controls";

type Account = { equity: number; cash: number; gross_exposure: number; day_pnl: number; day_pnl_pct: number };
type Status = { status: string; tier: string; halted_reason: string; trading_mode: string };
type Position = { symbol: string; qty: number; book: string; entry: number; mark: number; upnl: number; stop: number | null };

const pillColor: Record<string, string> = {
  RUNNING: "bg-emerald-700",
  PAUSED: "bg-amber-600",
  HALTED: "bg-red-700",
  STOPPED: "bg-gray-700",
};

export default function Dashboard() {
  const [account, setAccount] = useState<Account | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const { latest } = useStream(["equity", "positions", "engine_status"]);

  useEffect(() => {
    api<Account>("/account").then(setAccount).catch(() => {});
    api<Status>("/engine/status").then(setStatus).catch(() => {});
    api<Position[]>("/positions").then(setPositions).catch(() => {});
  }, []);

  const liveEq = latest.equity as { ts: string; equity: number; gross: number } | undefined;
  const livePos = latest.positions as { positions: Position[] } | undefined;
  const liveStatus = latest.engine_status as { status: string; reason?: string } | undefined;

  useEffect(() => {
    if (liveEq && account)
      setAccount({ ...account, equity: liveEq.equity, gross_exposure: liveEq.gross });
    if (livePos) setPositions(livePos.positions ?? []);
    if (liveStatus && status)
      setStatus({ ...status, status: liveStatus.status, halted_reason: liveStatus.reason ?? "" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latest]);

  const s = status?.status ?? "…";
  return (
    <main className="mx-auto max-w-6xl space-y-4 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">
          LOB Trading Platform{" "}
          <span className="ml-2 rounded bg-sky-900 px-2 py-0.5 text-xs text-sky-300">PAPER</span>
        </h1>
        <div className="flex items-center gap-3">
          {status && <TierSelector current={status.tier} />}
          <KillButton />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Card label="Equity" value={account ? `$${account.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "…"} />
        <Card
          label="Day P&L"
          value={account ? `$${account.day_pnl.toFixed(0)} (${account.day_pnl_pct.toFixed(2)}%)` : "…"}
          tone={account ? (account.day_pnl >= 0 ? "pos" : "neg") : undefined}
        />
        <Card label="Gross exposure" value={account ? `$${account.gross_exposure.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "…"} />
        <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
          <div className="text-xs text-gray-500">Engine</div>
          <span
            className={`mt-1 inline-block rounded-full px-3 py-1 text-xs font-semibold text-white ${pillColor[s] ?? "bg-gray-700"}`}
            data-testid="status-pill"
          >
            {s}
          </span>
          {status?.halted_reason && <div className="mt-1 text-xs text-red-400">{status.halted_reason}</div>}
        </div>
      </div>

      <EquityChart liveEquity={liveEq} />

      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <h2 className="mb-3 text-sm font-semibold text-gray-300">Positions ({positions.length})</h2>
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-gray-500">
            <tr>
              <th>Symbol</th><th>Type</th><th>Qty</th><th>Entry → Mark</th><th>Unrealized</th><th>Stop</th>
            </tr>
          </thead>
          <tbody className="text-gray-200">
            {positions.map((p) => (
              <tr key={p.symbol} className="border-t border-gray-800">
                <td className="py-2 font-medium">{p.symbol}</td>
                <td>
                  <span className={`rounded px-1.5 py-0.5 text-xs font-semibold ${p.book === "xsec" ? "bg-violet-900 text-violet-300" : "bg-sky-900 text-sky-300"}`}>
                    {p.book === "xsec" ? "Monthly" : "Day Trade"}
                  </span>
                </td>
                <td>{Math.abs(p.qty)}</td>
                <td>${p.entry?.toFixed(2)} → ${p.mark?.toFixed(2)}</td>
                <td className={p.upnl >= 0 ? "text-emerald-400" : "text-red-400"}>${p.upnl?.toFixed(2)}</td>
                <td>{p.stop ? `$${p.stop.toFixed(2)}` : "—"}</td>
              </tr>
            ))}
            {positions.length === 0 && (
              <tr>
                <td colSpan={5} className="py-4 text-center text-gray-600">flat</td>
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
