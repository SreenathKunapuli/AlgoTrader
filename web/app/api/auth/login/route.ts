// Login proxy: exchanges the password for a backend JWT and stores it in an
// httpOnly cookie — the token is never exposed to client-side JS.
import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_INTERNAL_BASE ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const res = await fetch(`${API}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    return NextResponse.json(await res.json().catch(() => ({})), {
      status: res.status,
    });
  }
  const { token } = await res.json();
  const out = NextResponse.json({ ok: true });
  out.cookies.set("lob_jwt", token, {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24,
  });
  return out;
}
