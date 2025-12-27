/*
 * clients/web/app/api/v1/billing/portal/route.ts
 *
 * Minimal Next.js App Router GET route that returns a deterministic billing portal URL
 * for the frontend "Manage billing" action. This endpoint intentionally reads only
 * NEXT_PUBLIC_* environment variables so it requires no server-only secrets.
 *
 * See: clients/web/API_CONTRACT.md (billing.portal -> { url: string })
 */

import { NextResponse } from 'next/server';

/**
 * Response shape returned by this route.
 */
type BillingPortalResponse = {
  url: string;
};

/**
 * GET /api/v1/billing/portal
 *
 * Behavior (deterministic):
 * - If NEXT_PUBLIC_BILLING_PORTAL_URL is set, return it.
 * - Else if NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is set, return a Stripe-like
 *   stub URL that includes the publishable key (safe to publish in browser).
 * - Otherwise return a harmless dev fallback URL.
 *
 * Important: This handler does NOT use any server-only secret env vars (e.g. STRIPE_SECRET_KEY).
 */
export async function GET(): Promise<Response> {
  const publicPortal = process.env.NEXT_PUBLIC_BILLING_PORTAL_URL;
  const publishable = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY;

  let url: string;

  if (publicPortal && publicPortal.trim().length > 0) {
    url = publicPortal.trim();
  } else if (publishable && publishable.trim().length > 0) {
    // deterministic, safe-to-expose stub URL that the frontend can open
    url = `https://billing.stripe.com/session_mock?pk=${encodeURIComponent(publishable.trim())}`;
  } else {
    // Safe dev fallback for local development or CI tests
    url = 'https://example.com/mock-billing-portal?dev=true';
  }

  const body: BillingPortalResponse = { url };
  return NextResponse.json(body, { status: 200 });
}

/*
Usage examples (curl):

$ curl -s http://localhost:3000/api/v1/billing/portal | jq
{
  "url": "https://example.com/mock-billing-portal?dev=true"
}

Front-end callers should expect JSON: { "url": string } and open that URL in a new tab/window.
*/
