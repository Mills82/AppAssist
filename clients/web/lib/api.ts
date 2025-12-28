// clients/web/lib/api.ts
// Typed centralized API client for frontend to call local /api/v1/* route handlers.
// Exports auth, usage, billing and project functions with token persistence and simple error handling.

import type {
  SessionToken,
  User,
  UsageResponse,
  BillingCheckoutResponse,
  Project,
  ImportJob,
  Credentials,
  ApiErrorBody,
  ProjectCreatePayload,
  CheckoutParams,
  BillingPortalResponse,
  AuthResponse
} from "./types";

/** Key used for localStorage persistence of the session token. */
const STORAGE_KEY = "app:session_token";
let inMemoryToken: string | null = null;

/**
 * ApiError is the canonical error type thrown by apiFetch and related helpers.
 * It always exposes a stable normalized shape accessible via normalized/normalizeApiError:
 *   { status: number, message: string, body?: unknown }
 * Conventions:
 *   - status = 0 indicates a network failure (no HTTP response received).
 *   - status > 0 maps to HTTP status codes when available.
 */
export class ApiError extends Error {
  status: number;
  body?: ApiErrorBody | unknown;

  constructor(message: string, status = 500, body?: ApiErrorBody | unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }

  /** Return a stable normalized object that UI code can rely on. */
  get normalized(): { status: number; message: string; body?: unknown } {
    return { status: this.status, message: this.message, body: this.body };
  }
}

/**
 * Extract a human-friendly message from various server response shapes.
 * Returns null when no message-like field can be determined.
 */
function extractMessageFromBody(body: unknown): string | null {
  if (body == null) return null;
  if (typeof body === "string") return body;
  if (typeof body === "object") {
    const b = body as any;
    if (typeof b.message === "string" && b.message.trim()) return b.message;
    if (typeof b.detail === "string" && b.detail.trim()) return b.detail;
    if (typeof b.error === "string" && b.error.trim()) return b.error;
    if (typeof b.error === "object" && b.error?.message) return String(b.error.message);
    if (typeof b.msg === "string" && b.msg.trim()) return b.msg;
    // fallback to a short JSON string for debugging, but limit length
    try {
      const s = JSON.stringify(body);
      return s.length > 0 ? (s.length > 1000 ? s.slice(0, 1000) + "..." : s) : null;
    } catch (_e) {
      return null;
    }
  }
  return null;
}

/**
 * Helper to normalize any thrown error into a stable {status, message, body?} shape.
 * Callers can use this to display consistent UI error messages without probing unknown shapes.
 */
export function normalizeApiError(err: unknown): { status: number; message: string; body?: unknown } {
  if (err instanceof ApiError) return err.normalized;
  if (err instanceof Error) return { status: 0, message: err.message || "Network request failed", body: undefined };
  // unknown non-Error value
  return { status: 0, message: String(err ?? "Unknown error"), body: err };
}

function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof localStorage !== "undefined";
}

export function getToken(): string | null {
  if (isBrowser()) {
    try {
      return localStorage.getItem(STORAGE_KEY) || null;
    } catch (_e) {
      // localStorage may be unavailable (private mode), fall back to memory
      return inMemoryToken;
    }
  }
  return inMemoryToken;
}

export function setToken(token: string | null) {
  inMemoryToken = token;
  if (isBrowser()) {
    try {
      if (token === null) localStorage.removeItem(STORAGE_KEY);
      else localStorage.setItem(STORAGE_KEY, token);
    } catch (_e) {
      // ignore localStorage write errors
    }
  }
}

export function clearToken() {
  setToken(null);
}

type FetchOpts = Omit<RequestInit, "body"> & { body?: unknown };

function normalizeRoute(path: string): string {
  // Ensure exactly one leading slash.
  return path.startsWith("/") ? path : `/${path}`;
}

function readCookie(name: string): string | null {
  // Cookie access must be guarded for SSR.
  if (!isBrowser() || typeof document === "undefined") return null;
  try {
    const raw = document.cookie
      .split(";")
      .map((s) => s.trim())
      .find((s) => s.startsWith(`${name}=`));
    return raw ? raw.split("=").slice(1).join("=") : null;
  } catch (_e) {
    return null;
  }
}

function buildAuthHeaders(existing?: HeadersInit): Record<string, string> {
  const headers: Record<string, string> = {
    ...(existing as Record<string, string> | undefined),
  };

  const token = getToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
    return headers;
  }

  // Fallback for dev stubs that may store session token in cookies.
  // We avoid overwriting Authorization if present, and only send this when no stored token.
  const cookieToken = readCookie(STORAGE_KEY);
  if (cookieToken && !headers["Authorization"] && !headers["authorization"]) {
    headers["x-session-token"] = cookieToken;
  }

  return headers;
}

function buildApiV1Url(path: string): string {
  const baseRaw = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
  const base = baseRaw.replace(/\/$/, "");
  const route = normalizeRoute(path);

  // If unset, rely on Next.js routing/rewrite and keep URL relative.
  if (!base) return `/api/v1${route}`;

  // If base already includes /api/v1, don't append it again.
  const prefix = base.endsWith("/api/v1") ? base : `${base}/api/v1`;
  return `${prefix}${route}`;
}

function defaultCredentialsForBase(baseRaw: string | undefined, existing?: RequestCredentials): RequestCredentials {
  // Relative URLs should send same-origin cookies (dev stubs + Next rewrites).
  // Absolute API bases should omit credentials by default to avoid accidental cross-site cookie use.
  const base = (baseRaw ?? "").replace(/\/$/, "");
  return existing ?? (base ? "omit" : "same-origin");
}

async function apiFetch<T = unknown>(path: string, opts: FetchOpts = {}): Promise<T> {
  const baseRaw = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
  const url = buildApiV1Url(path);

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...buildAuthHeaders(opts.headers),
  };

  const fetchOpts: RequestInit = {
    method: opts.method ?? (opts.body ? "POST" : "GET"),
    headers,
    credentials: defaultCredentialsForBase(baseRaw, opts.credentials),
    // allow callers to pass other fetch options (mode, cache, signal, etc.)
    ...(opts as RequestInit),
  };

  if (opts.body !== undefined) {
    try {
      fetchOpts.body = typeof opts.body === "string" ? opts.body : JSON.stringify(opts.body);
    } catch (e) {
      throw new ApiError("Failed to serialize request body", 400, e as Error);
    }
  }

  let res: Response;
  try {
    res = await fetch(url, fetchOpts);
  } catch (err) {
    // Network/TypeError failures should be normalized so UI can distinguish from HTTP errors.
    throw new ApiError((err as Error)?.message || "Network request failed", 0, err);
  }

  let text = "";
  try {
    text = await res.text();
  } catch (_e) {
    text = "";
  }

  let json: unknown = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch (_e) {
    // non-json response; keep raw text
    json = text;
  }

  if (!res.ok) {
    // Try to surface a meaningful message from server-provided JSON when possible
    const extracted = extractMessageFromBody(json);
    const msg = extracted ?? `${res.status} ${res.statusText}`;
    throw new ApiError(msg, res.status, json as ApiErrorBody | unknown);
  }

  return json as T;
}

/** Local Billing/Usage types used by the frontend. These are intentionally local so the UI can evolve independently of backend stubs. */
export type BillingTransaction = {
  id: string;
  amountCents: number;
  description?: string;
  createdAt: string; // ISO timestamp
  metadata?: Record<string, unknown>;
};

export type BillingResponse = {
  balanceCents: number;
  currency: string;
  transactions: BillingTransaction[];
};

/**
 * ===== SSE / Run stream helpers =====
 *
 * We use fetch + ReadableStream rather than EventSource so we can attach Authorization headers.
 * The server is expected to expose something like: GET /api/v1/runs/:runId/stream
 * and accept ?last_event_id=... for resuming.
 */

type SSEFrame = {
  id?: string;
  event?: string;
  data: string;
};

// Canonical SSEEvent / RunStatePatch types emitted by subscribeToRun.
export type SSEMessageEvent = {
  type: "message";
  id?: string;
  // payload from the server for a message (could be a structured object)
  message: unknown;
};

export type SSEStageEvent = {
  type: "stage";
  id?: string;
  stage: unknown;
};

export type SSEProgressEvent = {
  type: "progress";
  id?: string;
  stageId?: string;
  progress?: number;
  data?: unknown;
};

export type SSEErrorEvent = {
  type: "error";
  id?: string;
  error: { message?: string; code?: string } | unknown;
};

export type SSEDoneEvent = {
  type: "done";
  id?: string;
  reason?: string;
};

export type SSEEvent = SSEMessageEvent | SSEStageEvent | SSEProgressEvent | SSEErrorEvent | SSEDoneEvent;

// RunStatePatch is the normalized envelope consumers (RunState accumulator) should accept.
export type RunStatePatch = SSEEvent;

export type RunStreamCallbacks = {
  onPatch: (patch: RunStatePatch) => void;
  onError?: (err: Error) => void;
  onDone?: () => void;
};

export type RunStreamController = {
  stop: () => void;
  reconnect: () => void;
};

function escapeQueryValue(value: string): string {
  try {
    return encodeURIComponent(value);
  } catch (_e) {
    return value;
  }
}

/**
 * Parses a growing text buffer into complete SSE frames and returns any remainder.
 * Implements a minimal subset of the SSE spec.
 */
function parseSSEFrames(buffer: string): { frames: SSEFrame[]; rest: string } {
  // Normalize CRLF to LF to simplify parsing
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");

  // If buffer doesn't end with a frame delimiter, the last chunk is incomplete
  const hasTrailingDelimiter = normalized.endsWith("\n\n");
  const completeParts = hasTrailingDelimiter ? parts.filter((p) => p.length > 0) : parts.slice(0, -1);
  const rest = hasTrailingDelimiter ? "" : parts[parts.length - 1] ?? "";

  const frames: SSEFrame[] = [];

  for (const part of completeParts) {
    const lines = part.split("\n");
    const frame: SSEFrame = { data: "" };

    for (const rawLine of lines) {
      const line = rawLine.trimEnd();
      if (!line || line.startsWith(":")) continue; // comment or blank

      const idx = line.indexOf(":");
      const field = idx === -1 ? line : line.slice(0, idx);
      // Per SSE spec, if there is a space after ':', strip it.
      const value = idx === -1 ? "" : line.slice(idx + 1).replace(/^ /, "");

      if (field === "id") frame.id = value;
      else if (field === "event") frame.event = value;
      else if (field === "data") frame.data = frame.data ? `${frame.data}\n${value}` : value;
      // ignore retry and unknown fields
    }

    // Always emit frame if there was any meaningful data
    if (frame.data !== "" || frame.event || frame.id) {
      frames.push(frame);
    }
  }

  return { frames, rest };
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch (_e) {
    return text;
  }
}

function dedupeKeyForFrame(frame: SSEFrame): string {
  // Prefer server-provided id for strict ordering/dedupe.
  // Fallback to event+data for stubs that don't provide ids.
  return frame.id ? `id:${frame.id}` : `ev:${frame.event ?? "message"}|${frame.data}`;
}

function normalizeEventName(raw?: string): string {
  // Normalize '.' to ':' so callers can send run.error or run:error equivalently.
  return (raw ?? "").replace(/\./g, ":").toLowerCase();
}

function mapRawEventToPatch(frame: SSEFrame): RunStatePatch | null {
  // Normalize the event name and parse payload into the canonical SSEEvent shapes.
  const ev = normalizeEventName(frame.event);
  const data = safeJsonParse(frame.data);

  if (ev === "message" || ev.startsWith("message:")) {
    return {
      type: "message",
      id: frame.id,
      message: data,
    } as RunStatePatch;
  }

  if (ev === "stage" || ev.startsWith("stage:")) {
    return {
      type: "stage",
      id: frame.id,
      stage: data,
    } as RunStatePatch;
  }

  if (ev === "progress" || ev.startsWith("progress:")) {
    // data may contain progress number + optional stageId
    const progressNum = typeof data === "object" && data != null && (data as any).progress !== undefined ? (data as any).progress : undefined;
    const stageId = typeof data === "object" && data != null ? (data as any).stageId : undefined;
    return {
      type: "progress",
      id: frame.id,
      stageId,
      progress: typeof progressNum === "number" ? progressNum : undefined,
      data,
    } as RunStatePatch;
  }

  if (ev === "error" || ev === "run:error") {
    // Ensure error events normalize to SSEErrorEvent so the accumulator can mark status='error'
    const errPayload = typeof data === "object" && data != null ? data : { message: String(data) };
    return {
      type: "error",
      id: frame.id,
      error: errPayload,
    } as RunStatePatch;
  }

  if (ev === "done" || ev === "run:done") {
    const reason = typeof data === "object" && data != null ? (data as any).reason ?? undefined : undefined;
    return {
      type: "done",
      id: frame.id,
      reason,
    } as RunStatePatch;
  }

  // Fallback: if event name is unknown, try to infer from payload shape or emit a message envelope.
  return {
    type: "message",
    id: frame.id,
    message: data,
  } as RunStatePatch;
}

function isDoneEvent(frame: SSEFrame): boolean {
  const ev = normalizeEventName(frame.event);
  return ev === "done" || ev === "end" || ev === "completed" || ev === "run:done";
}

function isErrorEvent(frame: SSEFrame): boolean {
  const ev = normalizeEventName(frame.event);
  return ev === "error" || ev === "run:error";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function computeBackoffMs(attempt: number): number {
  // Capped exponential backoff: 250ms, 500ms, 1s, 2s, 4s, 8s, 10s...
  const base = 250;
  const max = 10_000;
  const ms = Math.min(max, base * Math.pow(2, Math.max(0, attempt)));
  // small jitter to avoid thundering herd
  const jitter = Math.floor(Math.random() * 150);
  return ms + jitter;
}

/**
 * Subscribe to a run's SSE stream. Deduplicates frames across reconnects and supports resume via last_event_id.
 */
export function subscribeToRun(runId: string, callbacks: RunStreamCallbacks): RunStreamController {
  let stopped = false;
  let abortController: AbortController | null = null;
  let reconnectAttempt = 0;

  const seenEventKeys = new Set<string>();
  let lastSeenEventId: string | null = null;

  const start = async (mode: "initial" | "reconnect") => {
    if (stopped) return;

    // Cancel any in-flight stream before starting a new one.
    if (abortController) {
      abortController.abort();
    }

    abortController = new AbortController();

    const qs = lastSeenEventId ? `?last_event_id=${escapeQueryValue(lastSeenEventId)}` : "";
    const url = buildApiV1Url(`/runs/${escapeQueryValue(runId)}/stream${qs}`);

    const baseRaw = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
    const headers: Record<string, string> = {
      Accept: "text/event-stream",
      ...buildAuthHeaders(undefined),
    };

    try {
      const res = await fetch(url, {
        method: "GET",
        headers,
        signal: abortController.signal,
        cache: "no-store",
        credentials: defaultCredentialsForBase(baseRaw, undefined),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        const parsed = safeJsonParse(text);
        const extracted = extractMessageFromBody(parsed);
        const msg = extracted ?? `${res.status} ${res.statusText}`;
        throw new ApiError(`SSE request failed: ${msg}`, res.status, parsed);
      }

      if (!res.body) {
        throw new ApiError("SSE response body is not readable", 500);
      }

      // Successful connection resets reconnect attempt counter.
      reconnectAttempt = 0;

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (!stopped) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSSEFrames(buffer);
        buffer = parsed.rest;

        for (const frame of parsed.frames) {
          // Track last id for resume
          if (frame.id) lastSeenEventId = frame.id;

          // Dedupe across reconnects
          const key = dedupeKeyForFrame(frame);
          if (seenEventKeys.has(key)) continue;
          seenEventKeys.add(key);

          if (isErrorEvent(frame)) {
            // Always surface error to caller and end stream.
            const data = safeJsonParse(frame.data);
            const msg = typeof data === "string" ? data : (data as any)?.message ?? "Run stream error";
            callbacks.onError?.(new ApiError(msg, 500, data));
            // Emit a normalized error patch so RunState accumulator can set status='error' and end the run
            callbacks.onPatch({ type: "error", id: frame.id, error: data } as RunStatePatch);
            callbacks.onDone?.();
            stopped = true;
            try {
              reader.cancel();
            } catch (_e) {
              // ignore
            }
            return;
          }

          const patch = mapRawEventToPatch(frame);
          if (patch) callbacks.onPatch(patch);

          if (isDoneEvent(frame)) {
            // Emit done patch and notify onDone so callers can finalize run state
            callbacks.onDone?.();
            stopped = true;
            try {
              reader.cancel();
            } catch (_e) {
              // ignore
            }
            return;
          }
        }
      }

      // If we exit loop without explicit stop/done, treat as disconnect and consider reconnect.
      if (!stopped) {
        // If stream ended naturally, attempt a reconnect with backoff.
        const waitMs = computeBackoffMs(reconnectAttempt++);
        await sleep(waitMs);
        if (!stopped) {
          await start("reconnect");
        }
      }
    } catch (err) {
      if (stopped) return;
      // Abort is an expected stop path.
      if ((err as any)?.name === "AbortError") return;

      // Normalize network errors similarly to apiFetch (status=0 sentinel).
      const normalized = err instanceof ApiError ? err : new ApiError((err as Error)?.message || "Network request failed", 0, err);
      callbacks.onError?.(normalized);

      // Auto-reconnect for transient errors.
      const waitMs = computeBackoffMs(reconnectAttempt++);
      await sleep(waitMs);
      if (!stopped) {
        await start(mode === "initial" ? "reconnect" : "reconnect");
      }
    }
  };

  // Kick off the stream.
  void start("initial");

  return {
    stop: () => {
      stopped = true;
      if (abortController) abortController.abort();
    },
    reconnect: () => {
      if (stopped) return;
      // Force a reconnect immediately (no backoff) while preserving dedupe state.
      reconnectAttempt = 0;
      void start("reconnect");
    },
  };
}

/** Auth/session endpoints */
export async function newSession(): Promise<{ token: SessionToken; user: User }> {
  const data = await apiFetch<{ token: SessionToken | { token?: string } | string; user: User }>("/session/new");
  // Normalize token formats: some stubs may return a bare string token, others an object { token: string }.
  // Store a simple string token in local persistence (localStorage / in-memory) so callers can rely on a string token.
  const raw = (data as any).token;
  const tokenString = typeof raw === "string" ? raw : raw?.token;
  setToken(tokenString ?? null);
  return data as { token: SessionToken; user: User };
}

export async function mockAuth(credentials: Credentials): Promise<{ token: SessionToken; user: User }> {
  const data = await apiFetch<{ token: SessionToken | { token?: string } | string; user: User }>("/auth/mock", {
    method: "POST",
    body: credentials,
  });
  // Normalize token formats similarly to newSession()
  const raw = (data as any).token;
  const tokenString = typeof raw === "string" ? raw : raw?.token;
  setToken(tokenString ?? null);
  return data as { token: SessionToken; user: User };
}

/** Convenience: postConversation - start a conversation / run from the frontend. */
export async function postConversation(body: unknown): Promise<unknown> {
  // Expose a simple API that callers (ChatWindow) expect when starting a run.
  return apiFetch<unknown>("/conversations", { method: "POST", body });
}

/** Usage endpoints */
export async function getUsage(simulate = false): Promise<UsageResponse> {
  const qs = simulate ? "?simulate=true" : "";
  return apiFetch<UsageResponse>(`/usage${qs}`);
}

export async function simulateUsage(): Promise<UsageResponse> {
  // instructs the backend stub to generate a synthetic usage event
  return apiFetch<UsageResponse>("/usage/simulate", { method: "POST" });
}

/** Billing / Stripe checkout */
export async function createCheckout(params: CheckoutParams): Promise<BillingCheckoutResponse> {
  // If a publishable key is present in env, prefer calling the backend to create a Stripe Checkout session.
  const hasStripe = !!process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY;
  if (hasStripe) {
    return apiFetch<BillingCheckoutResponse>("/billing/checkout", {
      method: "POST",
      body: params,
    });
  }

  // Stubbed flow for local development without Stripe configured
  const stub: BillingCheckoutResponse = {
    sessionId: `stub-checkout-${Date.now()}`,
    success: true,
    url: `/app/billing?mock_checkout=${Date.now()}`,
  } as BillingCheckoutResponse;
  // Optionally notify local backend so transactions show up in the dev UI
  try {
    await apiFetch<void>("/billing/checkout", { method: "POST", body: { ...params, _stub: true } });
  } catch (_e) {
    // ignore errors from optional stub notification
  }
  return stub;
}

export async function openBillingPortal(): Promise<BillingPortalResponse> {
  const hasStripe = !!process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY;
  if (hasStripe) {
    return apiFetch<BillingPortalResponse>("/billing/portal");
  }
  // Stubbed portal response
  return {
    url: "/app/billing?mock_portal=1",
  } as BillingPortalResponse;
}

/** Fetch billing summary: balance + recent transactions. */
export async function getBilling(): Promise<BillingResponse> {
  // backend stub should respond to GET /api/v1/billing with { balanceCents, currency, transactions: [...] }
  return apiFetch<BillingResponse>("/billing");
}

/** Convenience: fetch only numeric balance (calls getBilling internally). */
export async function getBalance(): Promise<{ balanceCents: number; currency: string }> {
  const b = await getBilling();
  return { balanceCents: b.balanceCents, currency: b.currency };
}

/** Projects */
export async function listProjects(): Promise<Project[]> {
  return apiFetch<Project[]>("/projects");
}

export async function createProject(payload: ProjectCreatePayload): Promise<Project> {
  return apiFetch<Project>("/projects", { method: "POST", body: payload });
}

export async function importProject(repoUrl: string): Promise<ImportJob> {
  return apiFetch<ImportJob>("/projects/import", { method: "POST", body: { repoUrl } });
}

// Backward-compat alias: some callers expect getProjects. Prefer canonical listProjects name
// but export the alias so both usages work during migration.
export const getProjects = listProjects;

// Minimal exports for convenience
export default {
  getToken,
  setToken,
  clearToken,
  subscribeToRun,
  postConversation,
  newSession,
  mockAuth,
  getUsage,
  simulateUsage,
  createCheckout,
  openBillingPortal,
  getBilling,
  getBalance,
  listProjects,
  createProject,
  importProject,
};

// Small runnable example for local dev: set environment variable RUN_API_EXAMPLE=true to run.
// This block executes only in Node (not in browser) and only when explicitly enabled.
if (typeof process !== "undefined" && process.env && process.env.RUN_API_EXAMPLE === "true") {
  (async () => {
    console.log("Running clients/web/lib/api.ts example...\n");
    try {
      console.log("Calling /session/new (expect stubbed session)");
      const s = await newSession();
      console.log("session:", s);

      console.log("Calling mockAuth with test credentials");
      const auth = await mockAuth({ email: "test@example.com", password: "password" });
      console.log("auth:", auth);

      console.log("Fetching usage (simulate=false)");
      const usage = await getUsage(false);
      console.log("usage:", usage);

      console.log("Simulating usage event (dev-only)");
      const usage2 = await simulateUsage();
      console.log("usage after simulate:", usage2);

      console.log("Creating stub checkout");
      const checkout = await createCheckout({ amountCents: 500, currency: "usd" });
      console.log("checkout:", checkout);

      console.log("Listing projects");
      const projects = await listProjects();
      console.log("projects:", projects);

      console.log("Fetching billing summary");
      const billing = await getBilling();
      console.log("billing:", billing);

      console.log("Done.");
    } catch (err) {
      console.error("Example run failed:", err);
      process.exitCode = 1;
    }
  })();
}
