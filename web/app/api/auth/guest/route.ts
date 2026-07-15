// Guest login: no password — issues a read-only token and sets the httpOnly cookie.
import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_INTERNAL_BASE ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const res = await fetch(`${API}/auth/guest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // forward the client IP for the rate limiter
    body: JSON.stringify({}),
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
    maxAge: 60 * 60 * 8,   // 8h matches token expiry
  });
  // Non-httpOnly role cookie so client JS can hide/show controls without an API call.
  out.cookies.set("lob_role", "guest", {
    httpOnly: false,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 8,
  });
  return out;
}
