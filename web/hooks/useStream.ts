"use client";
// Single WebSocket manager: subscribes to the requested channels, exposes the
// latest message per channel, auto-reconnects with capped backoff.
import { useEffect, useRef, useState } from "react";

export type StreamMsg = { channel: string; data: unknown; ts?: string };

export function useStream(channels: string[]) {
  const [latest, setLatest] = useState<Record<string, unknown>>({});
  const [connected, setConnected] = useState(false);
  const backoff = useRef(1000);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let timer: ReturnType<typeof setTimeout>;

    async function connect() {
      try {
        const r = await fetch("/api/ws-token");
        if (!r.ok) return;
        const { token } = await r.json();
        const base = process.env.NEXT_PUBLIC_WS_BASE ?? "ws://localhost:8000";
        ws = new WebSocket(`${base}/ws/stream?token=${token}`);
        ws.onopen = () => {
          setConnected(true);
          backoff.current = 1000;
          ws!.send(JSON.stringify({ subscribe: channels }));
        };
        ws.onmessage = (ev) => {
          const msg: StreamMsg = JSON.parse(ev.data);
          if (msg.channel && msg.channel !== "ping") {
            setLatest((prev) => ({ ...prev, [msg.channel]: msg.data }));
          }
        };
        ws.onclose = () => {
          setConnected(false);
          if (!closed) {
            timer = setTimeout(connect, backoff.current);
            backoff.current = Math.min(backoff.current * 2, 30000);
          }
        };
      } catch {
        if (!closed) timer = setTimeout(connect, backoff.current);
      }
    }
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      ws?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(channels)]);

  return { latest, connected };
}
