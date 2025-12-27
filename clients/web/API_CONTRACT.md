# API contract (local dev)

Purpose
-------
Defines the minimal HTTP endpoints used by the clients/web Next.js scaffold during local development. The contract documents endpoint paths, request/response shapes, SSE behavior, headers, and simple curl examples so frontend and backend agree on JSON shapes for local mocks.

Base URL
--------
- Default BACKEND_URL for local development: http://localhost:8080
- Frontend dev server: http://localhost:3000
- Note: the frontend expects the API to be mounted under /api/v1. For local development you can set BACKEND_URL to include /api/v1 (e.g. http://localhost:8080/api/v1) or set NEXT_PUBLIC_API_BASE_URL in the Next.js env to include the /api/v1 prefix.

CORS / dev note
---------------
The backend should permit requests from http://localhost:3000 during development (Access-Control-Allow-Origin). For local-dev stubs the backend may accept permissive CORS; do not use permissive CORS in production.

Also ensure the backend's FRONTEND_ORIGIN / allowed origins align with the frontend dev origin (recommended: http://localhost:3000) so cookies and preflight requests behave as expected in development.

Auth & headers (dev placeholders)
---------------------------------
- Authorization: Bearer <token> (optional placeholder)
- X-User: <user-id> (optional dev header)
- Content-Type: application/json

API Endpoints
-------------
All endpoints are rooted under /api/v1 for local development.

## GET /api/v1/ui/config
Purpose: Fetch UI feature flags and defaults used to render/enable client features.

Method: GET
Headers: optional Authorization

Response: 200 OK
Example response JSON:
```json
{
  "features": { "allowSignup": true, "modes": ["auto", "manual"] },
  "defaults": { "mode": "auto" }
}
```

## POST /api/v1/session/new
Purpose: Create a local mock session. The UI calls this to create/get a lightweight session object for dev flows.

Method: POST
Headers: Content-Type: application/json

Request example:
```json
{ "user_email": "dev@example.com" }
```

Response: 201 Created
Example response JSON:
```json
{
  "session_id": "sess_abc123",
  "user": { "id": "u1", "email": "dev@example.com", "name": "Dev" },
  "expires_at": "2026-01-01T00:00:00Z"
}
```

## POST /api/v1/auth/mock
Purpose: Return a mock auth token and user info for local-dev authentication flows (non-production).

Method: POST
Headers: Content-Type: application/json

Request example:
```json
{ "provider": "mock", "user_id": "u1" }
```

Response: 200 OK
Example response JSON:
```json
{ "token": "mock-token-xyz", "user": { "id": "u1", "email": "dev@example.com" } }
```

## POST /api/v1/conversation
Purpose: Conversation endpoint for intent/debug. The UI posts a user turn and receives an immediate mock response and metadata. This is useful to iterate on UI flows without the full engine.

Method: POST
Headers: Content-Type: application/json

Request example:
```json
{
  "session_id": "sess_abc123",
  "text": "Please summarize this file.",
  "metadata": { "project_id": "proj1" }
}
```

Response: 200 OK
Example response JSON:
```json
{
  "conversation_id": "conv_1",
  "messages": [
    { "role": "user", "text": "Please summarize this file." },
    { "role": "assistant", "text": "Summary: Lorem ipsum dolor sit amet..." }
  ],
  "debug": { "intent": "summarize" }
}
```

Notes: The UI expects an array of messages and an id the client can use to fetch or stream updates.

## POST /api/v1/llm
Purpose: Lightweight LLM proxy for summary/diff tasks in dev. This endpoint can be used to perform quick summarization, diff, or transform operations without streaming.

Method: POST
Headers: Content-Type: application/json

Request example:
```json
{
  "session_id": "sess_abc123",
  "prompt": "Summarize changes between A and B",
  "options": { "mode": "summary" }
}
```

Response: 200 OK
Example response JSON:
```json
{ "result": "Short summary...", "type": "summary" }
```

## GET /api/v1/events  (SSE)
Purpose: EventSource stream for UI realtime events (progress updates, logs, status). The UI connects via EventSource and receives newline-delimited event frames.

Connect example in the browser:
```js
const es = new EventSource(`${BACKEND_URL}/api/v1/events?session_id=sess_abc123`);
es.addEventListener('message', (e) => {
  const payload = JSON.parse(e.data);
  console.log('event', payload);
});
```

Query parameters:
- session_id (recommended): the session this stream is scoped to.

Transport: text/event-stream. Each SSE message should be sent as one or more "data: <JSON>" lines followed by a blank line. Example envelope (literal shown with backslash escapes to indicate line breaks):

```
data: {"type":"progress","id":"evt1","timestamp":"2025-01-01T12:00:00Z","payload":{"status":"started"}}\n\n
```

Sample event payload (JSON):
```json
{ "type": "progress", "id": "evt1", "timestamp": "2025-01-01T12:00:00Z", "payload": { "status": "started" } }
```

Client notes:
- EventSource auto-reconnects by default; optionally include a retry or let the client re-open on error.
- Send periodic heartbeats (e.g. `{ "type": "heartbeat" }`) during long-running tasks so clients can show liveness.
- Finalize with a `{ "type": "done", "id": "evtN" }` event when a job completes.

Examples (curl)
---------------
Fetch UI config:

```bash
curl "http://localhost:8080/api/v1/ui/config"
```

Create a session:

```bash
curl -X POST http://localhost:8080/api/v1/session/new -H 'Content-Type: application/json' -d '{"user_email":"dev@example.com"}'
```

Post a conversation turn:

```bash
curl -X POST http://localhost:8080/api/v1/conversation -H 'Content-Type: application/json' -d '{"session_id":"sess_abc123","text":"Please summarize this file.","metadata":{"project_id":"proj1"}}'
```

Stream events (follow output):

```bash
curl -N "http://localhost:8080/api/v1/events?session_id=sess_abc123"
```

Notes for implementers
----------------------
- These endpoints are intended as local-dev stubs. Mock tokens and responses are not secure and must not be treated as production credentials.
- Keep field names and JSON shapes in sync with `aidev/routes/frontend.py` and `aidev/server.py` CORS settings.
- Keep all paths under `/api/v1` to avoid clashes with other routes.
- For production, replace the mock auth/session/llm behavior with real implementations and tighten CORS and auth.
- Update this document whenever the backend shapes change so frontend and backend remain aligned.

