"use client";
import {
  createChart,
  IChartApi,
  ISeriesApi,
  LineData,
  LineSeries,
  UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

type Range = "1d" | "1w" | "1m" | "all";
const RANGES: Range[] = ["1d", "1w", "1m", "all"];

export default function EquityChart({ liveEquity }: { liveEquity?: { ts: string; equity: number } }) {
  const holder = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Line"> | null>(null);
  const lastTime = useRef<number>(0); // newest plotted ts; guards out-of-order updates
  const [range, setRange] = useState<Range>("1d");

  useEffect(() => {
    if (!holder.current) return;
    chart.current = createChart(holder.current, {
      height: 280,
      layout: { background: { color: "transparent" }, textColor: "#9ca3af" },
      grid: { vertLines: { color: "#1f2937" }, horzLines: { color: "#1f2937" } },
      timeScale: { timeVisible: true, secondsVisible: false },
    });
    series.current = chart.current.addSeries(LineSeries, { color: "#22c55e", lineWidth: 2 });
    const onResize = () =>
      chart.current?.applyOptions({ width: holder.current?.clientWidth ?? 600 });
    onResize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.current?.remove();
    };
  }, []);

  useEffect(() => {
    api<{ ts: string; equity: number }[]>(`/equity-curve?range=${range}`)
      .then((pts) => {
        // dedupe on second granularity and enforce ascending order
        const byTime = new Map<number, number>();
        for (const p of pts) byTime.set(Math.floor(new Date(p.ts).getTime() / 1000), p.equity);
        const data = Array.from(byTime.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([t, v]) => ({ time: t as UTCTimestamp, value: v }));
        series.current?.setData(data as LineData[]);
        lastTime.current = data.length ? (data[data.length - 1].time as number) : 0;
      })
      .catch(() => {});
  }, [range]);

  useEffect(() => {
    if (!liveEquity || !series.current) return;
    const t = Math.floor(new Date(liveEquity.ts).getTime() / 1000);
    if (t < lastTime.current) return; // stale tick from a load/stream race
    lastTime.current = t;
    series.current.update({ time: t as UTCTimestamp, value: liveEquity.equity });
  }, [liveEquity]);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-300">Equity</h2>
        <div className="flex gap-1">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`rounded px-2 py-1 text-xs ${
                r === range ? "bg-emerald-700 text-white" : "bg-gray-800 text-gray-400"
              }`}
            >
              {r.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      <div ref={holder} />
    </div>
  );
}
