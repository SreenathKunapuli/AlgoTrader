"use client";
// Tier selector (confirm dialog with limits) + kill switch (typed FLATTEN).
import { useState } from "react";
import { api } from "@/lib/api";
import { useRole } from "@/lib/useRole";

const TIER_LIMITS: Record<string, string> = {
  low: "5% max position · 40% gross · 1% daily loss kill · long only · daily rebalance",
  medium: "10% max position · 80% gross · 2% daily loss kill · long only · 15-min rebalance",
  high: "20% max position · 150% gross · 4% daily loss kill · long+short · 5-min rebalance",
};

export function TierSelector({ current }: { current: string }) {
  const role = useRole();
  const [pending, setPending] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (role === "guest") {
    return (
      <div className="flex items-center gap-2">
        <span className="rounded-lg border border-gray-700 bg-gray-900 px-3 py-1.5 text-xs font-medium capitalize text-gray-400">
          {current} tier
        </span>
        <span className="text-xs text-gray-600">(read-only)</span>
      </div>
    );
  }

  async function confirm() {
    if (!pending) return;
    setBusy(true);
    try {
      await api(`/config/tier`, { method: "PUT", body: JSON.stringify({ tier: pending }) });
      window.location.reload();
    } finally {
      setBusy(false);
      setPending(null);
    }
  }

  return (
    <div className="flex items-center gap-2">
      <div className="flex overflow-hidden rounded-lg border border-gray-700" data-testid="tier-selector">
        {["low", "medium", "high"].map((t) => (
          <button
            key={t}
            onClick={() => t !== current && setPending(t)}
            className={`px-3 py-1.5 text-xs font-medium capitalize ${
              t === current ? "bg-emerald-700 text-white" : "bg-gray-900 text-gray-400 hover:bg-gray-800"
            }`}
          >
            {t}
          </button>
        ))}
      </div>
      {pending && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" data-testid="tier-dialog">
          <div className="w-96 rounded-xl border border-gray-700 bg-gray-900 p-6">
            <h3 className="mb-2 font-semibold text-white">Switch to {pending.toUpperCase()} tier?</h3>
            <p className="mb-4 text-sm text-gray-400">{TIER_LIMITS[pending]}</p>
            <p className="mb-4 text-xs text-gray-500">Takes effect at the next rebalance.</p>
            <div className="flex justify-end gap-2">
              <button onClick={() => setPending(null)} className="rounded bg-gray-800 px-3 py-1.5 text-sm text-gray-300">
                Cancel
              </button>
              <button onClick={confirm} disabled={busy} className="rounded bg-emerald-700 px-3 py-1.5 text-sm text-white">
                {busy ? "…" : "Confirm"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function KillButton() {
  const role = useRole();
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);

  if (role === "guest") return null;

  async function fire() {
    setBusy(true);
    try {
      await api(`/engine/kill`, { method: "POST", body: JSON.stringify({ confirm: true }) });
      window.location.reload();
    } finally {
      setBusy(false);
      setOpen(false);
    }
  }

  return (
    <>
      <button
        onClick={() => { setTyped(""); setOpen(true); }}
        className="rounded-lg bg-red-700 px-4 py-1.5 text-sm font-semibold text-white hover:bg-red-600"
        data-testid="kill-button"
      >
        KILL SWITCH
      </button>
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" data-testid="kill-dialog">
          <div className="w-96 rounded-xl border border-red-900 bg-gray-900 p-6">
            <h3 className="mb-2 font-semibold text-red-400">Fire the kill switch?</h3>
            <p className="mb-4 text-sm text-gray-400">
              Cancels all open orders and flattens every position with market orders,
              then HALTS the engine until an explicit reset.
            </p>
            <p className="mb-2 text-xs text-gray-500">Type <b>FLATTEN</b> to confirm:</p>
            <input
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              className="mb-4 w-full rounded border border-gray-700 bg-black px-2 py-1.5 text-sm text-white"
              data-testid="kill-confirm-input"
            />
            <div className="flex justify-end gap-2">
              <button onClick={() => setOpen(false)} className="rounded bg-gray-800 px-3 py-1.5 text-sm text-gray-300">
                Cancel
              </button>
              <button
                onClick={fire}
                disabled={typed !== "FLATTEN" || busy}
                className="rounded bg-red-700 px-3 py-1.5 text-sm text-white disabled:opacity-40"
                data-testid="kill-confirm-button"
              >
                {busy ? "…" : "FLATTEN ALL"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
