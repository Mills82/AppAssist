/*
 * clients/web/app/api/v1/billing/checkout/route.ts
 *
 * Dev-friendly Next.js App Router POST route for billing checkout.
 * Returns a mock Stripe Checkout URL when NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is present,
 * otherwise returns a deterministic dev stub session object.
 */

import { NextResponse } from "next/server";

// Allow using `require.main` in a TS file without type errors in most setups
declare const require: any;

/** Request shape accepted by this endpoint. */
type BillingCheckoutRequest = {
  amount?: number; // in cents (e.g. 1000 = $10.00)
  currency?: string;
  description?: string;
  projectId?: string;
};

/** Response shape when a Stripe-like URL is returned. */
type BillingCheckoutUrlResponse = { url: string };

/** Response shape used for local/dev stub checkout. */
type BillingDevSessionResponse = { sessionId: string; success: true };

/** Union response type for the endpoint. */
export type BillingCheckoutResponse = BillingCheckoutUrlResponse | BillingDevSessionResponse;

/**
 * Small pure helper that composes the response object based on presence
 * of a publishable key. Kept pure so it can be unit-tested outside Next.
 */
export function createCheckoutResponse(
  body: BillingCheckoutRequest = {},
  publishableKey?: string
): BillingCheckoutResponse {
  // Validate amount if present
  if (body.amount !== undefined) {
    if (typeof body.amount !== "number" || Number.isNaN(body.amount) || body.amount <= 0) {
      // Throw a sentinel error that the route handler will map to 400
      throw new Error("invalid_amount");
    }
  }

  if (publishableKey) {
    // Construct a deterministic-ish mock checkout URL that includes the publishable key.
    const url = `https://checkout.stripe.com/pay/${encodeURIComponent(publishableKey)}_stub_${Date.now()}`;
    return { url };
  }

  // Dev fallback: return a simple session object so the frontend can continue the flow.
  return { sessionId: `dev_stub_${Date.now()}`, success: true };
}

/**
 * Next.js App Router POST handler.
 * - Accepts JSON body matching BillingCheckoutRequest
 * - Validates amount when provided
 * - Returns { url } when NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY is set
 * - Otherwise returns { sessionId, success: true }
 *
 * Note: Real Stripe integration should happen on a secure server using the
 * secret Stripe key (not publishable) and the official Stripe SDK. This route
 * is intentionally self-contained and does NOT call Stripe or require secret env vars.
 */
export async function POST(req: Request) {
  try {
    const body = (await req.json().catch(() => ({}))) as BillingCheckoutRequest;

    const publishable = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY;

    const resp = createCheckoutResponse(body, publishable);

    return NextResponse.json(resp, { status: 200 });
  } catch (err: any) {
    if (err instanceof Error && err.message === "invalid_amount") {
      return NextResponse.json({ error: "invalid amount" }, { status: 400 });
    }

    // Unexpected server error
    // eslint-disable-next-line no-console
    console.error("/api/v1/billing/checkout error:", err);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }
}

/*
  Local/dev usage example (can be run with ts-node or used in unit tests).
  This example exercises the pure helper and demonstrates both branches.

  Example:
    npx ts-node clients/web/app/api/v1/billing/checkout/route.ts
*/
if (typeof require !== "undefined" && require.main === module) {
  (async () => {
    // Dev branch (no publishable key)
    const dev = createCheckoutResponse({ amount: 500 });
    // Prod-like branch (publishable key present)
    const prod = createCheckoutResponse({ amount: 1500 }, "pk_test_demo_12345");

    // eslint-disable-next-line no-console
    console.log("Dev response:", dev);
    // eslint-disable-next-line no-console
    console.log("Prod (mock URL) response:", prod);
  })();
}
