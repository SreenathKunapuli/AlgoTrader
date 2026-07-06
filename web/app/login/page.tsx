"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Login() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const router = useRouter();

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const r = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    setBusy(false);
    if (r.ok) router.push("/");
    else setError(r.status === 429 ? "Too many attempts — wait a minute." : "Invalid password.");
  }

  return (
    <main className="flex min-h-screen items-center justify-center">
      <form onSubmit={submit} className="w-80 space-y-4 rounded-xl border border-gray-800 bg-gray-900 p-8">
        <h1 className="text-lg font-bold text-white">LOB Platform</h1>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          data-testid="password-input"
          className="w-full rounded border border-gray-700 bg-black px-3 py-2 text-sm text-white"
        />
        {error && <p className="text-xs text-red-400">{error}</p>}
        <button
          type="submit"
          disabled={busy || !password}
          data-testid="login-button"
          className="w-full rounded bg-emerald-700 py-2 text-sm font-semibold text-white disabled:opacity-40"
        >
          {busy ? "…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
