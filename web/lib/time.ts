// Display edge: all UI times render in America/New_York per the platform spec.
export function fmtET(ts: string | number | Date, withDate = false): string {
  const d = new Date(ts);
  return d.toLocaleString("en-US", {
    timeZone: "America/New_York",
    ...(withDate ? { month: "short", day: "numeric" } : {}),
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  }) + " ET";
}

// lightweight-charts labels its axis from raw UTC epochs; shift so the axis
// reads as ET wall-clock (DST-aware via Intl).
export function toEtEpoch(ts: string): number {
  const d = new Date(ts);
  const et = new Date(d.toLocaleString("en-US", { timeZone: "America/New_York" }));
  return Math.floor(et.getTime() / 1000);
}
