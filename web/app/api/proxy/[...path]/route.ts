// Catch-all REST proxy: forwards /api/proxy/* to the FastAPI backend with
// the JWT from the httpOnly cookie as a Bearer header.
import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_INTERNAL_BASE ?? "http://localhost:8000";

async function forward(req: NextRequest, path: string[]) {
  const token = req.cookies.get("lob_jwt")?.value;
  const url = `${API}/${path.join("/")}${req.nextUrl.search}`;
  const res = await fetch(url, {
    method: req.method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: ["GET", "HEAD"].includes(req.method) ? undefined : await req.text(),
    cache: "no-store",
  });
  const data = await res.text();
  return new NextResponse(data, {
    status: res.status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function GET(req: NextRequest, ctx: { params: { path: string[] } }) {
  return forward(req, ctx.params.path);
}
export async function POST(req: NextRequest, ctx: { params: { path: string[] } }) {
  return forward(req, ctx.params.path);
}
export async function PUT(req: NextRequest, ctx: { params: { path: string[] } }) {
  return forward(req, ctx.params.path);
}
