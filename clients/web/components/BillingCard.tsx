import React, { useCallback, useMemo, useState } from "react";

// clients/web/components/BillingCard.tsx
// Reusable billing card: shows balance, last-updated, and primary billing actions.

export interface BillingCardProps {
  balance: number; // in major currency units, e.g. 12.34
  currency?: string; // ISO currency code, default 'USD'
  lastUpdated?: string | Date;
  onAddFunds?: () => Promise<void> | void;
  onManageBilling?: () => void;
  loading?: boolean; // external loading state
}

export default function BillingCard({
  balance,
  currency = "USD",
  lastUpdated,
  onAddFunds,
  onManageBilling,
  loading = false,
}: BillingCardProps) {
  const [localLoading, setLocalLoading] = useState(false);

  const isDisabled = loading || localLoading;

  const formatted = useMemo(() => {
    try {
      return new Intl.NumberFormat(undefined, {
        style: "currency",
        currency,
      }).format(balance ?? 0);
    } catch (e) {
      // Fallback
      return `${currency} ${Number(balance ?? 0).toFixed(2)}`;
    }
  }, [balance, currency]);

  const lastUpdatedLabel = useMemo(() => {
    if (!lastUpdated) return null;
    try {
      const d = typeof lastUpdated === "string" ? new Date(lastUpdated) : lastUpdated;
      return d instanceof Date && !isNaN(d.getTime()) ? d.toLocaleString() : String(lastUpdated);
    } catch (e) {
      return String(lastUpdated);
    }
  }, [lastUpdated]);

  // Handler: prefer caller-supplied onAddFunds; otherwise attempt to call lib/api.createCheckout.
  const handleAddFunds = useCallback(async () => {
    if (isDisabled) return;
    setLocalLoading(true);
    try {
      if (onAddFunds) {
        await onAddFunds();
        return;
      }

      // Try dynamic imports so this component doesn't hard-fail at compile-time
      // if api client path differs in some projects. We try two common import paths.
      let createCheckout: (() => Promise<any>) | undefined;
      try {
        // prefer path alias if available
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const mod = await import("lib/api");
        createCheckout = mod?.createCheckout ?? mod?.default?.createCheckout;
      } catch (e) {
        try {
          const mod = await import("../lib/api");
          createCheckout = mod?.createCheckout ?? mod?.default?.createCheckout;
        } catch (e2) {
          createCheckout = undefined;
        }
      }

      if (typeof createCheckout === "function") {
        // createCheckout should either redirect (by returning a URL or using window.location)
        // or return a stubbed result. We await it to show loading state while it resolves.
        await createCheckout();
        // Caller will likely refresh balance via outer state; we don't mutate balance here.
        // But we can show a small success toast by using alert as minimal feedback.
        // In the real app, replace with toast system.
        try {
          // small UX: if checkout returned without redirect, inform dev
          // (createCheckout implementations should handle redirect when using Stripe)
          // no-op here
        } catch (e) {
          /* ignore */
        }
      } else {
        // No API available: perform stubbed dev flow (simulate adding $10)
        await new Promise((res) => setTimeout(res, 700));
        // Minimal feedback; in real app you'd refresh server-side balance.
        // We call window.alert only in dev-like flows to make the action visible.
        if (typeof window !== "undefined") {
          // eslint-disable-next-line no-alert
          window.alert("Stubbed add funds complete (dev). Please refresh to see updated balance.");
        }
      }
    } catch (err) {
      // Basic error feedback; replace with toasts in the app
      if (typeof window !== "undefined") {
        // eslint-disable-next-line no-console
        console.error("Add funds failed", err);
        // eslint-disable-next-line no-alert
        window.alert("Failed to initiate checkout. See console for details.");
      }
    } finally {
      setLocalLoading(false);
    }
  }, [isDisabled, onAddFunds]);

  const handleManageBilling = useCallback(() => {
    if (onManageBilling) return onManageBilling();
    // Fallback stub: open Stripe customer portal would normally be done server-side.
    if (typeof window !== "undefined") {
      // eslint-disable-next-line no-alert
      window.alert("Manage billing (stub): open Stripe customer portal from server in production.");
    }
  }, [onManageBilling]);

  return (
    <div className="p-4 bg-white rounded shadow-sm flex flex-col gap-3 w-full max-w-md">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-sm text-gray-500">Account balance</div>
          <div
            data-testid="billing-balance"
            aria-label={`Account balance: ${formatted}`}
            className="mt-1 text-2xl font-semibold text-gray-900"
          >
            {formatted}
          </div>
        </div>
        <div className="text-right text-xs text-gray-400">
          {lastUpdatedLabel ? <div>Updated</div> : null}
          {lastUpdatedLabel ? <div className="mt-1 text-gray-500">{lastUpdatedLabel}</div> : null}
        </div>
      </div>

      <div className="flex gap-2">
        <button
          data-testid="btn-add-funds"
          type="button"
          onClick={handleAddFunds}
          disabled={isDisabled}
          className={`inline-flex items-center justify-center px-4 py-2 rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed text-sm`}
          aria-disabled={isDisabled}
        >
          {localLoading ? "Processing..." : "Add funds"}
        </button>

        <button
          data-testid="btn-manage-billing"
          type="button"
          onClick={handleManageBilling}
          disabled={isDisabled}
          className="inline-flex items-center justify-center px-3 py-2 rounded-md text-sm border border-gray-200 text-gray-700 hover:bg-gray-50 disabled:opacity-60 disabled:cursor-not-allowed"
        >
          Manage billing
        </button>
      </div>

      <div className="text-xs text-gray-400">Payments handled by Stripe. Your card details are never stored here.</div>
    </div>
  );
}

// Small demo export to help developers render the component in isolation.
// Example usage in Storybook or a dev page:
export function BillingCardDemo() {
  const [bal, setBal] = useState(42.5);
  const onAddFunds = async () => {
    // simulate a server-side checkout completing and adding funds
    await new Promise((r) => setTimeout(r, 600));
    setBal((s) => Number((s + 10).toFixed(2)));
    if (typeof window !== "undefined") {
      // eslint-disable-next-line no-alert
      window.alert("Demo: +$10 added to balance (local demo only)");
    }
  };
  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <BillingCard balance={bal} currency="USD" lastUpdated={new Date().toISOString()} onAddFunds={onAddFunds} />
    </div>
  );
}

