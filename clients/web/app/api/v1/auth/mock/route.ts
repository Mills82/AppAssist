// clients/web/app/api/v1/auth/mock/route.ts
// DEV-STUB: lightweight in-process auth mock for frontend development/testing.
// Accepts POST { email, password } and returns { token, user } for allowed test creds.

import { NextResponse } from "next/server";
import type { User, AuthResponse } from "../../../../../lib/types";

// In-memory session store for the running dev server process.
// Other dev-only route handlers may import this module to read session state.
export const DEV_SESSIONS = new Map<string, { user: User; createdAt: number }>();

function makeToken(email: string) {
  // Deterministic token so tests and frontend can rely on predictable values.
  // NOTE: Buffer is available in Node runtime used by Next.js route handlers.
  return `dev:${Buffer.from(email).toString("base64")}`;
}

function makeUser(email: string): User {
  const local = String(email).split("@")[0] || email;
  const name = local.replace(/[^a-zA-Z0-9._-]/g, " ") || "Dev User";
  return {
    id: `user:${email}`,
    email,
    name,
  };
}

// POST handler: accept { email, password } and return { token, user }.
export async function POST(req: Request) {
  try {
    const body = (await req.json()) as Record<string, unknown> | undefined;
    const email = typeof body?.email === "string" ? body.email.trim() : undefined;
    const password = typeof body?.password === "string" ? body.password : undefined;

    if (!email || !password) {
      return NextResponse.json({ error: "Missing email or password" }, { status: 400 });
    }

    // Dev acceptance rules:
    // - any email ending with `@example.com` OR
    // - password equals the magic dev password `password`
    // These are intentionally permissive for local development only.
    const allowed = email.endsWith("@example.com") || password === "password";

    if (!allowed) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
    }

    const user = makeUser(email);
    const token = makeToken(email);

    // Persist session in-memory so other dev route stubs can look it up.
    DEV_SESSIONS.set(token, { user, createdAt: Date.now() });

    const resp: AuthResponse = { token, user };
    return NextResponse.json(resp);
  } catch (err) {
    return NextResponse.json({ error: "Invalid JSON payload" }, { status: 400 });
  }
}

// Disallow GET (clarifies supported methods). Next.js will route GET here if called.
export async function GET() {
  return new Response(JSON.stringify({ error: "Method Not Allowed" }), {
    status: 405,
    headers: { "Content-Type": "application/json", Allow: "POST" },
  });
}

