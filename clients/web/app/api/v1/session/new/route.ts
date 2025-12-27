/*
  clients/web/app/api/v1/session/new/route.ts
  Dev-only stub route: returns a deterministic session token and user payload.
  Acceptance: GET or POST -> 200 JSON { token: string, user: { id: string, email: string, name?: string } }
  POST may accept { email?: string, name?: string } and will reflect provided email/name in the returned user.
  NOTE: This is an in-memory, side-effect-free stub. Replace with real auth backend in production.
*/

import { NextResponse } from "next/server";

// Local types to keep this file self-contained. Consider centralizing into clients/web/lib/types.ts later.
interface User {
  id: string;
  email: string;
  name?: string;
}

interface SessionResponse {
  token: string;
  user: User;
}

const DEFAULT_TOKEN = "dev-session-token";
const DEFAULT_USER_ID = "dev-user-1";
const DEFAULT_EMAIL = "dev@example.com";
const DEFAULT_NAME = "Dev User";

function makeSession(email?: string, name?: string): SessionResponse {
  return {
    token: DEFAULT_TOKEN,
    user: {
      id: DEFAULT_USER_ID,
      email: email ?? DEFAULT_EMAIL,
      name: name ?? DEFAULT_NAME,
    },
  };
}

/**
 * GET /api/v1/session/new
 * Returns a deterministic dev session payload.
 */
export async function GET(_request: Request) {
  const payload = makeSession();
  return NextResponse.json(payload, { status: 200 });
}

/**
 * POST /api/v1/session/new
 * Accepts JSON body { email?: string, name?: string } and returns a session reflecting provided values.
 * This does NOT authenticate or persist anything; it's purely a dev stub to bootstrap frontend sessions.
 */
export async function POST(request: Request) {
  try {
    const body = (await request.json?.()) as Partial<{ email: string; name: string }> | undefined;
    const payload = makeSession(body?.email, body?.name);
    return NextResponse.json(payload, { status: 200 });
  } catch (err) {
    // If request.json() fails (malformed body), return a simple default session instead of an error to keep dev flows smooth.
    const payload = makeSession();
    return NextResponse.json(payload, { status: 200 });
  }
}
