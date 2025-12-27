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

- Backend default port: 8000. The frontend expects the Python backend to expose the UI API router under /api/v1. Use the single authoritative example for the frontend base URL:

  export NEXT_PUBLIC_API_BASE_URL="http://localhost:8000/api/v1"

  Windows PowerShell:

  $env:NEXT_PUBLIC_API_BASE_URL = 'http://localhost:8000/api/v1'

  Note: NEXT_PUBLIC_API_BASE_URL should include the /api/v1 path used by the stub routes.

Local development

This project includes a development-time proxy so frontend developers can call /api/v1/* directly without changing the backend origin in most cases.

1. Install dependencies

   npm install

2. Start the Python backend (local dev / FastAPI)

   - Default backend URL: http://localhost:8000
   - From the repository root (example):
     python -m aidev.server

   - Ensure FRONTEND_ORIGIN is set before starting the server (see Backend / CORS configuration above) so the backend will add CORS allowance for http://localhost:3000.
   - The frontend router used by the UI is mounted at /api/v1 when the FastAPI router is present; the frontend client expects the API at that prefix.

3. Point the frontend at the backend (optional when using the dev proxy)

   - If you prefer to set an explicit base URL instead of using the dev proxy, set NEXT_PUBLIC_API_BASE_URL to the backend URL including /api/v1:

     - macOS / Linux / WSL / Git Bash:
       export NEXT_PUBLIC_API_BASE_URL="http://localhost:8000/api/v1"

     - Windows PowerShell:
       $env:NEXT_PUBLIC_API_BASE_URL = 'http://localhost:8000/api/v1'

   Note: NEXT_PUBLIC_API_BASE_URL should include the /api/v1 path used by the stub routes.

4. Start the Next.js dev server

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

Ensure clients/web/package.json contains the scripts above; if missing, add them so npm run dev/build/start/lint work as documented.

Verify
- Open http://localhost:3000 and confirm public pages render: /, /pricing, /docs, /about, /contact.
- Visit http://localhost:3000/app/dashboard to verify the chat-first dashboard loads with the Sidebar, Project selector, Approvals badge, Recent Runs, Chat window and ModeChip (Auto/Manual toggle). Advanced controls are collapsed by default.

API Contract & Backend Stubs
- See clients/web/API_CONTRACT.md for the documented frontend↔backend contract (endpoints, example payloads, and SSE URL).
- The Python backend exposes lightweight stub endpoints for local development (aidev/routes/frontend.py). These include UI config, mock session/auth, conversation and LLM proxies, and an EventSource-compatible /api/v1/events stream stub.
- Ensure the Python backend allows CORS from the frontend origin (http://localhost:3000) while developing locally by setting FRONTEND_ORIGIN before starting the FastAPI server.

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

Project layout (where to look)
- clients/web/app        — Next.js App Router pages (public + /app)
- clients/web/components — Shared React components (Sidebar, ModeChip, ChatWindow, etc.)
- clients/web/package.json — npm scripts (dev, build, start)
- clients/web/API_CONTRACT.md — Frontend↔backend API contract and examples

Notes for contributors
- This README is intentionally short. Expand environment, docker-compose, or CI/dev notes as needed.
- Do not commit secrets or production credentials to this folder. Use environment variables for configuration.

If something is unclear or a stub endpoint is missing, check clients/web/API_CONTRACT.md and aidev/routes/frontend.py for the expected request/response shapes.

---

Frontend-local API stubs and required env vars

This repository now includes optional frontend-local Next.js Route Handler stubs under clients/web/app/api/v1/ so you can run the UI without the Python backend. When the Next dev server is running (npm run dev) these route handlers will respond to /api/v1/* requests. They are intended for local development only and use in-memory state where applicable (projects, imports, usage simulation).

Required env vars (local development)

- Create a .env.local at clients/web/.env.local (ignored by git) and add (examples only — DO NOT commit real keys):
  NEXT_PUBLIC_API_BASE_URL="http://localhost:8000/api/v1" # optional when dev proxy active
  NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_test_XXXX" # optional; leave unset to use stubbed checkout

- Notes:
  - NEXT_PUBLIC_API_BASE_URL: If you want the frontend to call an external backend (e.g., the Python server), set this to that backend's base URL including /api/v1. When the Next dev server dev-proxy is active you can call /api/v1/* directly and do not need to set this.
  - NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY: When set, the billing checkout endpoint will return a Stripe checkout URL in the stub response to exercise the real flow. If unset, the checkout endpoint responds with a deterministic dev-session response so the UI can complete without calling Stripe.

- Dev-time proxy behavior

- During development clients/web/next.config.js rewrites /api/v1/* to http://localhost:8000/api/v1 (the Python backend) when NODE_ENV !== 'production'. If you rely solely on the Next route stubs included in this frontend, you can call /api/v1/* without setting NEXT_PUBLIC_API_BASE_URL. If you want to point at another backend, set NEXT_PUBLIC_API_BASE_URL to e.g. http://localhost:8000/api/v1.

Typed client + types

- A small typed API client is provided at clients/web/lib/api.ts and TypeScript types at clients/web/lib/types.ts. Prefer importing and using these instead of calling fetch directly from pages/components so token handling and base URL logic stays centralized.

Quick curl checks for the new frontend route stubs (run against http://localhost:3000 when next dev is running)

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

- Projects import (GitHub flow - stubbed)
  curl -sS -X POST http://localhost:3000/api/v1/projects/import -H 'Content-Type: application/json' -d '{"repo":"owner/repo"}' | jq
  Expected: { "id": "import-<id>", "repo": "owner/repo", "status": "imported" }

- Notes
- In-memory state for projects/imports is ephemeral and reset when the Next dev server restarts; restart dev server to clear stub state.
- The typed client lives at clients/web/lib/api.ts and types at clients/web/lib/types.ts — prefer these exports to keep fetch wiring consistent and to centralize token handling.
- If you change contract shapes, update clients/web/API_CONTRACT.md accordingly. The frontend stubs aim to match that contract; any additions are documented here.

+ Auth (login / signup)
+
+ The frontend includes developer-friendly login and signup pages and a small client-side session model for local development. These pages/flows are implemented against the local mock endpoints under /api/v1 so they work without a real backend.
+
+ Pages
+
+ - /app/login — email + password form
+ - /app/signup — name, email + password form
+
+ Mock auth endpoints (dev stubs)
+
+ - POST /api/v1/auth/mock
+   Request JSON: { "email": "user@example.com", "password": "..." }
+   Response JSON: { "token": "<string>", "user": { "id": "<id>", "email": "user@example.com", "name": "Full Name" } }
+
+   Example curl:
+   curl -sS -X POST http://localhost:3000/api/v1/auth/mock -H 'Content-Type: application/json' -d '{"email":"test@example.com","password":"x"}' | jq
+
+ - GET /api/v1/session/new
+   Response JSON: { "token": "<string>", "user": { "id": "<id>", "email": "<email>", "name": "<name>" } }
+
+ Client-side session
+
+ - Session state is provided by an AuthProvider (clients/web/context/AuthProvider.tsx) and a typed hook useAuth (clients/web/lib/useAuth.ts).
+ - By default the AuthProvider persists the token and user in sessionStorage so the session survives full page reloads during development-only flows.
+ - UI components should use useAuth() to access: { user, token, signIn, signUp, signOut, loading }.
+
+ Route protection
+
+ - /app/* routes are guarded by the AuthProvider (see clients/web/app/layout.tsx). If a user is not authenticated they are redirected to /app/login. After successful auth the user is redirected back to their intended route.
+
+ Auth redirect query param
+
+ - The frontend uses the query parameter callbackUrl to record the user's intended destination when redirecting to /app/login (e.g., /app/login?callbackUrl=/app/dashboard). The AuthProvider and the login/signup pages read and honor callbackUrl when redirecting the user back after successful auth. Keep this param name consistent if you change the provider or pages.
+
+ Swapping to a real backend
+
+ - To replace the mock endpoints with a real backend:
+   - Update clients/web/lib/api.ts to point to your backend base URL (NEXT_PUBLIC_API_BASE_URL) and ensure auth routes follow the shapes above (or update API_CONTRACT.md to the new shapes).
+   - Remove or disable the Next route stubs under clients/web/app/api/v1/ that implemented the mock auth endpoints.
+   - Keep the AuthProvider/useAuth seam: the provider persists token+user and exposes signIn/signUp/signOut — swap the internals to call your real endpoints.
+
+ Verification checklist additions
+
+ - [ ] Visit /app/login and /app/signup and submit forms; successful auth stores token+user in sessionStorage and redirects to /app/dashboard.
+ - [ ] `curl -sS http://localhost:3000/api/v1/session/new` returns { token, user }.
+ - [ ] `curl -sS -X POST http://localhost:3000/api/v1/auth/mock ...` returns token+user.
+
Notes
- In-memory state for projects/imports is ephemeral and reset when the Next dev server restarts; restart dev server to clear stub state.
- The typed client lives at clients/web/lib/api.ts and types at clients/web/lib/types.ts — prefer these exports to keep fetch wiring consistent and to centralize token handling.
- If you change contract shapes, update clients/web/API_CONTRACT.md accordingly. The frontend stubs aim to match that contract; any additions are documented here.
- Mock endpoint locations: when using the frontend-only stubs the mock endpoints referenced in this README (for example /api/v1/auth/mock and /api/v1/session/new) are implemented as Next.js Route Handlers under clients/web/app/api/v1/. If you prefer to run the Python backend instead, those endpoints are provided by aidev/routes/frontend.py. Update this README if you change where the stubs live.

Verification checklist (maps to acceptance criteria)

- [ ] npm install && npm run dev starts the Next dev server successfully.
- [ ] GET /api/v1/session/new (curl) returns 200 and JSON { token, user } as shown above.
- [ ] POST /api/v1/auth/mock accepts test credentials and returns token + user JSON.
- [ ] GET /api/v1/usage returns sample events and summary; calling with ?simulate=true returns altered/demo data.
- [ ] POST /api/v1/billing/checkout returns a Stripe URL when NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is set, otherwise returns a stubbed dev session response.
- [ ] GET /api/v1/projects and POST /api/v1/projects behave and allow creating/listing projects in-memory for the session.


## Billing, Usage & Projects — Local testing

Quick UI steps to validate the new flows (use the Next dev server at http://localhost:3000):

1. Environment
   - Create clients/web/.env.local (DO NOT COMMIT). You can copy the example file and edit values:
     cp clients/web/.env.local.example clients/web/.env.local
   - Add or confirm the following entries in clients/web/.env.local (example values only):
     NEXT_PUBLIC_API_BASE_URL="http://localhost:8000/api/v1" # optional when dev proxy active
     NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_test_XXXX" # optional; leave unset to exercise the stubbed checkout flow
   - Behavior summary:
     - If NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is set, the billing checkout stub will return a Stripe Checkout URL so you can exercise the redirect-to-Stripe flow.
     - If NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is unset, the checkout endpoint will return a deterministic dev session and the UI will complete the flow without contacting Stripe.

2. /app/billing
   - Visit http://localhost:3000/app/billing. Expect to see a numeric balance and a recent transactions list, or a friendly empty state if there are no transactions.
   - Click "Add funds":
     - With a publishable key configured you may be redirected to Stripe Checkout (stubbed URL can be returned by the route handlers).
     - Without a publishable key the stubbed flow will complete immediately and the in-memory balance will increase so you can exercise UI updates.
   - Click "Manage billing" — this should open the billing portal stub URL returned by GET /api/v1/billing/portal.

3. /app/usage
   - Visit http://localhost:3000/app/usage to view usage events and a summary line showing totals and computed cost.
   - Developer-only: click the "Simulate usage" button in the UI (or call GET /api/v1/usage?simulate=true) to add an event. After simulation the events list and the shown balance should update to reflect higher usage/cost.

4. /app/projects
   - Visit http://localhost:3000/app/projects to view existing projects or an empty state.
   - Use the Create Project form (name + optional description) to create a project — the list should update immediately (in-memory).
   - Use the GitHub Connect button to exercise the stubbed import state machine:
     - Click → the button becomes "connected" → import is initiated → shows "importing" → when finished it shows "imported" and the imported repo appears in the projects list.
   - You can also exercise the import stub via POST /api/v1/projects/import as shown in the curl examples above.

5. Ephemeral state & reset
   - The frontend-only route handlers under clients/web/app/api/v1/ use in-memory state; this state is cleared when the Next dev server restarts. Restart the dev server to reset projects, usage simulations, and billing stubs.

Acceptance checklist mapping (quick):
- Visiting /app/billing shows numeric balance and transactions (or empty state).
- "Add funds" triggers checkout and either redirects to Stripe (if key present) or completes the stubbed flow and increases balance.
- /app/usage lists events; simulating usage updates events and reduces balance in the UI.
- /app/projects allows creating a project and the GitHub connect button moves through connected → importing → imported states and displays the imported repo.

If any endpoint shapes change, update clients/web/API_CONTRACT.md and clients/web/lib/api.ts accordingly.

Restarting Next dev server will reset in-memory stub state.

Follow-up
- The route handlers that implement these stubs live under clients/web/app/api/v1/. If you add or change endpoint payload shapes, update clients/web/API_CONTRACT.md and the typed client at clients/web/lib/api.ts & clients/web/lib/types.ts so pages keep working.
