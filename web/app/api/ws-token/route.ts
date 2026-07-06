// The browser needs the JWT to open the backend WebSocket (query param).
// This same-origin, cookie-gated endpoint hands it over only to an already
// authenticated session — an accepted single-user tradeoff (DECISIONS.md).
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const token = req.cookies.get("lob_jwt")?.value;
  if (!token) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  return NextResponse.json({ token });
}
