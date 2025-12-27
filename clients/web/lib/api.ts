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

export class ApiError extends Error {
  status: number;
  body?: ApiErrorBody | unknown;

  constructor(message: string, status = 500, body?: ApiErrorBody | unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
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

async function apiFetch<T = unknown>(path: string, opts: FetchOpts = {}): Promise<T> {
  const baseRaw = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
  const base = baseRaw.replace(/\/$/, "");
  const prefix = base ? `${base}/api/v1` : "/api/v1";
  const route = path.startsWith("/") ? path : `/${path}`;
  const url = `${prefix}${route}`;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(opts.headers as Record<string, string> | undefined),
  };

  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const fetchOpts: RequestInit = {
    method: opts.method ?? (opts.body ? "POST" : "GET"),
    headers,
    // allow callers to pass other fetch options (credentials, mode, etc.)
    ...(opts as RequestInit),
  };

  if (opts.body !== undefined) {
    try {
      fetchOpts.body = typeof opts.body === "string" ? opts.body : JSON.stringify(opts.body);
    } catch (e) {
      throw new ApiError("Failed to serialize request body", 400, e as Error);
    }
  }

  const res = await fetch(url, fetchOpts);
  const text = await res.text();
  let json: unknown = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch (_e) {
    // non-json response
    json = text;
  }

  if (!res.ok) {
    throw new ApiError(`Request failed: ${res.status} ${res.statusText}`, res.status, json as ApiErrorBody | unknown);
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
