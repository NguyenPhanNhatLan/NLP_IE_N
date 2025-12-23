import { NextResponse } from "next/server";

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(req: Request, ctx: Ctx) {
    const { path } = await ctx.params;
    return proxy(req, path);
}
export async function POST(req: Request, ctx: Ctx) {
    const { path } = await ctx.params;
    return proxy(req, path);
}
export async function PUT(req: Request, ctx: Ctx) {
    const { path } = await ctx.params;
    return proxy(req, path);
}
export async function PATCH(req: Request, ctx: Ctx) {
    const { path } = await ctx.params;
    return proxy(req, path);
}
export async function DELETE(req: Request, ctx: Ctx) {
    const { path } = await ctx.params;
    return proxy(req, path);
}

async function proxy(req: Request, pathParts: string[]) {
    const base = process.env.NLP_BACKEND_URL;
    if (!base) return NextResponse.json({ error: "NLP_BACKEND_URL not set" }, { status: 500 });

    const incomingUrl = new URL(req.url);
    const baseUrl = base.endsWith("/") ? base : base + "/";

    // backend của bạn có prefix /ie
    const target = new URL(baseUrl + "ie/" + pathParts.join("/"));
    target.search = incomingUrl.search;

    const headers = new Headers(req.headers);
    headers.delete("host");
    headers.delete("content-length");
    headers.delete("connection");
    headers.delete("accept-encoding");

    const init: RequestInit = {
        method: req.method,
        headers,
        cache: "no-store",
        redirect: "manual",
    };

    if (req.method !== "GET" && req.method !== "HEAD") {
        init.body = await req.arrayBuffer();
    }

    const r = await fetch(target, init);
    return new Response(r.body, { status: r.status, headers: r.headers });
}
