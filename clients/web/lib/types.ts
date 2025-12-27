// clients/web/lib/types.ts
// Centralized TypeScript types for the frontend API client and route handler stubs.
// See: clients/web/API_CONTRACT.md for the canonical contract this file mirrors.

/**
 * A short-lived session token returned by auth/session endpoints.
 * In runtime the server returns a plain token string, so keep this a string alias
 * to match the actual runtime shape and make it easy to swap for a richer object later.
 */
export type SessionToken = string;

/**
 * Minimal user profile shape used across the frontend.
 */
export interface User {
  /** Stable user id (UUID or opaque string) */
  id: string;
  /** Primary email address */
  email: string;
  /** Optional full name */
  name?: string;
  /** Optional avatar image URL */
  avatarUrl?: string;
  /** Account creation timestamp (ISO-8601) */
  createdAt?: string;
}

/**
 * Generic API error shape returned by mocked endpoints.
 */
export interface APIError {
  code?: string;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * A simple alias/type representing a parsed error response body from the API.
 * Route handlers and the API client may use this type when parsing error JSON.
 */
export type ApiErrorBody = APIError;

/**
 * Credentials payload used by auth endpoints (login/signup) and by tests/dev mocks.
 */
export interface Credentials {
  email: string;
  password: string;
}

/**
 * Auth response returned by auth/mock and session/new endpoints: token + user pair.
 */
export interface AuthResponse {
  token: SessionToken;
  user: User;
}

/**
 * Currency balance represented in minor units (cents) to avoid floating point issues.
 */
export interface Balance {
  currency: string; // e.g. "USD"
  amountCents: number; // integer cents (e.g. $1.23 -> 123)
  updatedAt: string; // ISO-8601
}

/**
 * Transaction representing a billing event (credit or debit).
 */
export interface Transaction {
  id: string;
  type: "credit" | "debit";
  amountCents: number;
  currency: string;
  description?: string;
  createdAt: string; // ISO-8601
}

/**
 * Usage event recorded by the system. Extended to include project/endpoint info
 * and per-event cost breakdown so the UI can render richer usage rows.
 */
export interface UsageEvent {
  id: string;
  timestamp: string; // ISO-8601
  projectId?: string;
  endpoint?: string; // logical endpoint or action name
  action?: string; // backward-compatible action field (e.g. "api_call", "model_run")
  count?: number; // number of units for this event (e.g. tokens, calls)
  unitCostCents?: number; // cost per unit in cents
  totalCostCents?: number; // computed cost for this event in cents
  meta?: Record<string, unknown>;
}

/**
 * Aggregated usage summary returned alongside event lists.
 */
export interface UsageSummary {
  totalCalls: number;
  totalCostCents: number;
  periodStart?: string; // ISO-8601
  periodEnd?: string; // ISO-8601
}

/**
 * Standard response envelope for usage endpoints.
 */
export interface UsageResponse {
  events: UsageEvent[];
  summary: UsageSummary;
}

/**
 * Response returned when creating/initiating a billing checkout session.
 * If running with real Stripe configuration the endpoint may return a url to redirect to.
 */
export interface BillingCheckoutResponse {
  /** Optional hosted checkout URL (Stripe Checkout session url) */
  url?: string;
  /** Optional provider session id (e.g. stripe session id) */
  sessionId?: string;
  /** Whether the server stub considered the request successful */
  success?: boolean;
  message?: string;
}

/**
 * Response for a billing portal (manage billing) request.
 */
export interface BillingPortalResponse {
  url: string; // URL to redirect customer to billing portal (stub or real)
  sessionId?: string;
  success?: boolean;
  message?: string;
}

/**
 * Possible states for an import job (GitHub/project import flow).
 */
export type ImportStatus = "idle" | "pending" | "importing" | "completed" | "failed";

/**
 * Tracks an import job for a project/repo import. Extended fields align the shape
 * with route handlers that may return importId, pollCount, sessionToken, result, etc.
 */
export interface ImportJob {
  id?: string;
  importId?: string;
  projectId?: string;
  repoUrl?: string;
  status: ImportStatus;
  pollCount?: number;
  createdAt?: string;
  startedAt?: string;
  finishedAt?: string;
  sessionToken?: string;
  message?: string; // human friendly status or error
  result?: Record<string, unknown> | null;
}

/**
 * Represents a project managed in the frontend MVP.
 */
export interface Project {
  id: string;
  name: string;
  description?: string;
  /** Optional linked repository URL (when imported/connected) */
  repoUrl?: string;
  imported?: boolean;
  importJobId?: string;
  createdAt?: string;
  /** Optional token identifying the owner/session that created/owns the project */
  ownerToken?: string;
}

/**
 * Payload used to create a new project via POST /projects
 */
export interface ProjectCreatePayload {
  name: string;
  description?: string;
  repoUrl?: string;
  ownerToken?: string;
}

/**
 * Parameters sent to initiate a checkout session (billing/checkout)
 */
export interface CheckoutParams {
  amountCents: number;
  currency?: string; // default e.g. "USD"
  returnUrl?: string;
  projectId?: string;
}

/**
 * Generic paginated response wrapper used by list endpoints.
 */
export interface Paginated<T> {
  items: T[];
  total: number;
  page: number;
  perPage: number;
}

// --- Run / SSE event types used by ChatWindow and RunTimeline ---
/** Known common SSE event.type values; include common variant spellings (colon and dot)
 *  so consumers can accept either form while migrating. Unknown event types are still allowed.
 */
export type KnownEventType =
  | 'run:stage'
  | 'run.stage'
  | 'run:progress'
  | 'run.progress'
  | 'run:error'
  | 'run.error'
  | 'message'
  | string;

/** Canonical envelope shape normalized from server-sent events (SSE).
 *  payload is intentionally any because different events carry different shapes;
 *  ts may be emitted as either an ISO string or a numeric unix/ms timestamp, so accept both.
 */
export interface EventEnvelope {
  type: KnownEventType;
  payload?: any;
  /** ISO-8601 timestamp or numeric epoch (ms) when the server emitted the event (optional). */
  ts?: string | number;
}

/** Status for an individual stage in a run timeline. */
export type StageStatus = 'pending' | 'active' | 'done' | 'error';

/** Representation of a stage for the RunTimeline UI. */
export interface StageState {
  id: string;             // stable stage id (backend-provided)
  title: string;          // human-friendly title (preferred canonical field)
  /**
   * Legacy alias: some components historically consumed `label` instead of `title`.
   * Keep this optional alias for backward compatibility during migration. Prefer `title`.
   */
  label?: string;
  status: StageStatus;    // pending | active | done | error
  progress?: number;      // optional 0-100 numeric progress for the stage
  startedAt?: string;     // ISO-8601
  finishedAt?: string;    // ISO-8601
  message?: string;       // optional human message or error text
  meta?: Record<string, unknown>;
}

/** Props consumed by a RunTimeline component. */
export interface RunTimelineProps {
  stages: StageState[];
  currentStageId?: string;
  progress?: number; // global run progress 0..100
  errors?: (string | { message: string; code?: string })[];
  otherEvents?: EventEnvelope[]; // secondary list of unknown/ignored events
}

// End of file
