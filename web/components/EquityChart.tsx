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
      .then((pts) =>
        series.current?.setData(
          pts.map((p) => ({
            time: (new Date(p.ts).getTime() / 1000) as UTCTimestamp,
            value: p.equity,
          })) as LineData[]
        )
      )
      .catch(() => {});
  }, [range]);

  useEffect(() => {
    if (liveEquity && series.current) {
      series.current.update({
        time: (new Date(liveEquity.ts).getTime() / 1000) as UTCTimestamp,
        value: liveEquity.equity,
      });
    }
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
