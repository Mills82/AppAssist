# AI Dev Bot — Web Frontend (clients/web)

A minimal Next.js (App Router) + TypeScript + Tailwind scaffold intended for local frontend development and integration with the AI Dev Bot Python backend. This folder contains the public marketing pages and a /app chat-first dashboard (slim sidebar, ModeChip, collapsible Advanced controls) used for UI development against backend stubs.

Prerequisites
- Node.js 16+ (18+ recommended)
- npm or yarn
- Python backend running for API stubs (see aidev/server.py)

Backend / CORS configuration (important for local dev)
- The Python backend needs to know the frontend origin to allow CORS for local development.
- Set FRONTEND_ORIGIN before starting the Python server so aidev/server.py can include the correct origin in its CORS configuration. Example (macOS / Linux / WSL / Git Bash):

  export FRONTEND_ORIGIN="http://localhost:3000"

  Windows PowerShell:

  $env:FRONTEND_ORIGIN = 'http://localhost:3000'

- Backend default port: 8000. The frontend expects the Python backend to expose the UI API router under the /api/v1 prefix (the typed client appends /api/v1 to a base URL; see env section below for precise behavior).

Local development

Dev modes (choose one)

This frontend supports two local dev modes. In both modes, frontend code should call API paths via the typed client at clients/web/lib/api.ts (it centralizes base URL resolution, credentials policy, and error normalization).

A) Frontend-only (Next.js Route Handler stubs)
- No FastAPI required.
- Next.js Route Handlers under clients/web/app/api/v1/ respond to /api/v1/* while npm run dev is running.
- Use this mode to develop UI flows with in-memory stub state (projects/imports/usage/billing).

B) Frontend + FastAPI via dev proxy (no code changes)
- Run the Python backend on http://localhost:8000.
- In development, clients/web/next.config.js rewrites /api/v1/* to http://localhost:8000/api/v1 when NODE_ENV !== 'production'.
- Use this mode when you want the UI to hit the real FastAPI server while keeping frontend calls as /api/v1/*.

Run frontend against FastAPI (no code changes)

1) Export CORS origin (before starting FastAPI)

   macOS / Linux / WSL / Git Bash:
   export FRONTEND_ORIGIN="http://localhost:3000"

   Windows PowerShell:
   $env:FRONTEND_ORIGIN = 'http://localhost:3000'

2) Start FastAPI (from repository root)

   python -m aidev.server

   - Default backend URL: http://localhost:8000
   - The UI router is expected at /api/v1 when the FastAPI router is present; the typed client will resolve the full URL as described below.

3) Start the Next.js dev server

   npm run dev

   - By default Next.js runs on http://localhost:3000.
   - In dev, next.config.js proxies /api/v1/* → http://localhost:8000/api/v1 (active when NODE_ENV !== 'production').

Required / optional env vars (local development)

Create a .env.local at clients/web/.env.local (ignored by git). Example values only — DO NOT commit real keys.

- NEXT_PUBLIC_API_BASE_URL (optional)
  - What: Override the API base URL for the typed client.
  - Canonical convention (important):
    - If NEXT_PUBLIC_API_BASE_URL is unset, the client uses a relative base and requests are made to /api/v1/* (this works with Next.js Route Handler stubs and when next.config.js rewrites /api/v1/* to a backend).
    - If NEXT_PUBLIC_API_BASE_URL is set, treat it as an origin/base (no required /api/v1). The client will append exactly one /api/v1 to that base unless the provided value already ends with /api/v1. This avoids accidental double-prefixing.
    - Examples (recommended):
      - Use a plain origin: NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"  # client will call http://localhost:8000/api/v1/...
      - If you prefer to include /api/v1 explicitly, that is supported, but not recommended as the default: NEXT_PUBLIC_API_BASE_URL="http://localhost:8000/api/v1"
  - Credentials/CORS implications (default policies):
    - Relative requests (NEXT_PUBLIC_API_BASE_URL unset): the client uses credentials='same-origin' so cookies (session) are sent to the same origin (works with Next.js stubs and rewrite proxy to backend).
    - Absolute base (NEXT_PUBLIC_API_BASE_URL set): by default the client uses credentials='omit' to avoid sending cookies cross-origin; callers can override this per-request where needed. This is deliberate to avoid unexpected cookie/CORS behavior when pointing the frontend at another origin.

- NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY (optional)
  - What: Enables returning a Stripe Checkout URL from the billing checkout stub so you can exercise the redirect flow.
  - When: Optional; leave unset to use deterministic stubbed checkout responses.
  - Example: NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_test_XXXX"

- NEXT_PUBLIC_BILLING_PORTAL_URL (optional; verify usage in codebase)
  - What: Optional override for a billing portal URL.
  - When: Only relevant if referenced by the UI/client; reconcile with clients/web/lib/api.ts and related pages before relying on it. The README lists it for convenience but callers should verify code usage.
  - Example: NEXT_PUBLIC_BILLING_PORTAL_URL="http://localhost:3000/billing/portal"

- NEXT_PUBLIC_ENABLE_DEV_ACTIONS (optional; verify usage in codebase)
  - What: Optional toggle for dev-only behaviors (e.g., showing simulate buttons).
  - When: Only relevant if referenced by the UI; reconcile with code before treating as authoritative.
  - Example: NEXT_PUBLIC_ENABLE_DEV_ACTIONS="true"

Example .env.local (clients/web/.env.local)

NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"  # optional; client appends /api/v1 unless already present
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_test_XXXX"
NEXT_PUBLIC_BILLING_PORTAL_URL="http://localhost:3000/billing/portal"  # optional
NEXT_PUBLIC_ENABLE_DEV_ACTIONS="true"  # optional, toggle dev-only behaviors

Notes on environment and dev proxy
- When running the Next dev server you can call /api/v1/* directly without setting NEXT_PUBLIC_API_BASE_URL because the typed client will use a relative base when the env var is unset. This covers both Next.js Route Handler stubs and the next.config.js dev rewrite to the Python backend.
- If you set NEXT_PUBLIC_API_BASE_URL to an absolute origin, the client will append /api/v1 (unless you already included it) and will default to credentials='omit' to avoid cross-origin cookie leaks. Adjust per-request when you need cross-origin credentials.

Install & run

1. Install dependencies

   npm install

2. Start the Python backend (local dev / FastAPI) [optional]

   - Default backend URL: http://localhost:8000
   - From the repository root (example):
     python -m aidev.server

   - Ensure FRONTEND_ORIGIN is set before starting the server (see Backend / CORS configuration above) so the backend will add CORS allowance for http://localhost:3000.
   - The frontend router used by the UI is mounted at /api/v1 when the FastAPI router is present; the typed client expects the API at that prefix.

3. Start the Next.js dev server

   npm run dev

   - By default Next.js runs on http://localhost:3000 — running npm run dev starts the Next.js dev server on that address.
   - When running the Next dev server, clients/web/next.config.js contains a dev-only rewrite that proxies /api/v1/* to http://localhost:8000/api/v1 so frontend code can call /api/v1/* directly without setting NEXT_PUBLIC_API_BASE_URL. This dev-only rewrite is active in development (i.e., when NODE_ENV !== 'production').

Helpful npm scripts

- npm run dev  — Start Next.js in development mode (default: http://localhost:3000)
- npm run build — Build the production bundle
- npm run start — Start the built Next.js app (after npm run build)
- npm run lint  — Run the project's linter

Package.json scripts (example)

```
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  }
}
```

Verify (quick checks)
- Open http://localhost:3000 and confirm public pages render: /, /pricing, /docs, /about, /contact.
- Visit http://localhost:3000/app/dashboard to verify the chat-first dashboard loads with the Sidebar, Project selector, Approvals badge, Recent Runs, Chat window and ModeChip (Auto/Manual toggle). Advanced controls are collapsed by default.
- When testing API wiring: with NEXT_PUBLIC_API_BASE_URL unset the client will call relative /api/v1/* (works with route stubs and dev proxy); when NEXT_PUBLIC_API_BASE_URL is set to an absolute origin the client appends /api/v1 and uses credentials='omit' by default.

API client, errors & types

- A small typed API client is provided at clients/web/lib/api.ts and TypeScript types at clients/web/lib/types.ts.
- The typed client centralizes base URL resolution, token handling, credentials policy, and error normalization — update it in lockstep with contract changes.
- Error normalization: the client throws an ApiError for non-2xx responses and network failures. ApiError exposes a stable surface: { status: number, message: string, body?: any } where status=0 is reserved for network failures (fetch/network error). The UI should catch ApiError and read status and message for consistent user-facing error display; body is an optional extra detail for advanced handling or logging.

Quick curl checks for the frontend route stubs (run against http://localhost:3000 when next dev is running)

- GET session/new
  curl -sS http://localhost:3000/api/v1/session/new | jq
  Expected: { "token": "<string>", "user": { "id": "<id>", "email": "<email>", "name": "<name>" } }

- POST auth/mock
  curl -sS -X POST http://localhost:3000/api/v1/auth/mock -H 'Content-Type: application/json' -d '{"email":"test@example.com","password":"x"}' | jq
  Expected: { "token": "<string>", "user": { "id": "<id>", "email": "test@example.com", "name": "Test User" } }

- GET usage (simulate)
  curl -sS "http://localhost:3000/api/v1/usage?simulate=true" | jq
  Expected: { "events": [ /* event objects */ ], "summary": { "totalCalls": <number>, "cost": <number> } }

- POST billing/checkout
  curl -sS -X POST http://localhost:3000/api/v1/billing/checkout -H 'Content-Type: application/json' -d '{}' | jq
  Expected: if NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is set: { "url": "https://checkout.stripe.com/..." }
            else: { "sessionId": "dev-session-<id>", "success": true }

- GET billing/portal (manage billing)
  curl -sS http://localhost:3000/api/v1/billing/portal | jq
  Expected: { "url": "https://billing-portal.example.dev/..." } or a dev stub URL.

- Projects list/create
  curl -sS http://localhost:3000/api/v1/projects | jq
  curl -sS -X POST http://localhost:3000/api/v1/projects -H 'Content-Type: application/json' -d '{"name":"My Project"}' | jq
  Expected: GET returns array of projects; POST returns created project with id and createdAt.

Notes
- In-memory state for projects/imports is ephemeral and reset when the Next dev server restarts; restart dev server to clear stub state.
- The typed client lives at clients/web/lib/api.ts and types at clients/web/lib/types.ts — prefer these exports to keep fetch wiring consistent and to centralize token handling.
- If you change contract shapes, update clients/web/API_CONTRACT.md accordingly. The frontend stubs aim to match that contract; any additions are documented there.
- The README lists NEXT_PUBLIC_BILLING_PORTAL_URL and NEXT_PUBLIC_ENABLE_DEV_ACTIONS for convenience; verify whether they are referenced in code (clients/web/lib/api.ts or app pages) and either remove from docs or wire them in code as appropriate.

TODO / Included public pages
- The scaffold expects the following public marketing pages to exist as simple placeholders under clients/web/app:
  - /
  - /pricing
  - /docs
  - /about
  - /contact
- If any of those pages are missing, add simple placeholder pages in clients/web/app so the header links and routing render without errors.

Troubleshooting
- "Network" or "CORS" errors in the browser devtools usually indicate NEXT_PUBLIC_API_BASE_URL is unset/misconfigured, FRONTEND_ORIGIN wasn't provided to the backend before it started, or the backend isn't running.
- Confirm the backend process is running on the expected port (default: 8000) and that API paths match the contract in clients/web/API_CONTRACT.md.
- If you change ports, update NEXT_PUBLIC_API_BASE_URL accordingly and restart the Next dev server and the Python backend.
- If you see CORS errors, confirm FRONTEND_ORIGIN was exported before starting the Python server.
- If you are using the Next dev server, verify that requests to /api/v1/* in your browser are being proxied to http://localhost:8000/api/v1 (this confirms the dev rewrite in next.config.js is active).

Verification checklist (maps to acceptance criteria)

- [ ] npm install && npm run dev starts the Next dev server successfully.
- [ ] GET /api/v1/session/new (curl) returns 200 and JSON { token, user } as shown above (frontend-only Next.js stubs).
- [ ] GET /api/v1/session/new returns 200 when FastAPI is running and the dev proxy rewrite is active (clients/web/next.config.js rewrites /api/v1/* → http://localhost:8000/api/v1).
- [ ] POST /api/v1/auth/mock accepts test credentials and returns token + user JSON.
- [ ] GET /api/v1/usage returns sample events and summary; calling with ?simulate=true returns altered/demo data.
- [ ] POST /api/v1/billing/checkout returns a Stripe URL when NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is set, otherwise returns a stubbed dev session response.
- [ ] GET /api/v1/projects and POST /api/v1/projects behave and allow creating/listing projects in-memory for the session.
- [ ] API errors: the client throws ApiError with a stable { status, message, body? } surface for UI consumption; status=0 is reserved for network failures. Ensure UI code catches ApiError and reads status/message for display.

Follow-up / notes for contributors
- This README is intentionally concise. Expand environment, docker-compose, or CI/dev notes as needed.
- Do not commit secrets or production credentials to this folder. Use environment variables for configuration.
- If you change the clients/web/lib/api.ts base URL resolution or error shape, update this README and clients/web/API_CONTRACT.md so contributors have a single authoritative source for dev wiring and error handling.

If something is unclear or a stub endpoint is missing, check clients/web/API_CONTRACT.md and aidev/routes/frontend.py for the expected request/response shapes.
