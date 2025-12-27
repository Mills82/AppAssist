/* clients/web/app/usage/page.tsx

Client-side Next.js page: shows usage events, current balance, and a dev-only "Simulate usage" action.
*/

"use client";

import React, { useEffect, useState } from "react";
import { getUsage, getBilling, simulateUsage } from "../../lib/api";

// Local types (move to shared types when available)
type UsageEvent = {
  id: string;
  amountCents: number;
  description?: string;
  createdAt: string;
};

type UsageResponse = { events: UsageEvent[] };
type BillingResponse = { balanceCents: number };
type SimulateResponse = { events: UsageEvent[]; balanceCents: number };

function formatCurrency(cents: number) {
  return "$" + (cents / 100).toFixed(2);
}

export default function UsagePage() {
  const [events, setEvents] = useState<UsageEvent[] | null>(null);
  const [balance, setBalance] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [simulating, setSimulating] = useState(false);

  // Dev-only toggle: set NEXT_PUBLIC_ENABLE_DEV_ACTIONS=true to expose simulate in non-prod builds
  const devActions =
    process.env.NEXT_PUBLIC_ENABLE_DEV_ACTIONS === "true" ||
    (process.env.NEXT_PUBLIC_NODE_ENV !== "production" && process.env.NODE_ENV !== "production");

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [u, b] = await Promise.all([getUsage(), getBilling()]);
      setEvents(u.events ?? []);
      setBalance(b.balanceCents ?? 0);
    } catch (err: any) {
      console.error(err);
      setError(err?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // Intentionally no dependencies: run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onSimulate() {
    setSimulating(true);
    setError(null);
    try {
      const res = await simulateUsage();
      setEvents(res.events ?? []);
      setBalance(res.balanceCents ?? null);
    } catch (err: any) {
      console.error(err);
      setError(err?.message ?? "Simulation failed");
    } finally {
      setSimulating(false);
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Usage</h1>
        <p className="text-sm text-gray-500">Recent usage events and account balance.</p>
      </header>

      <section className="mb-6">
        {/* TODO: replace with shared BillingCard component at clients/web/components/BillingCard.tsx */}
        <div className="bg-white shadow rounded p-4">
          <h2 className="text-sm text-gray-500">Current balance</h2>
          <div className="mt-2 text-3xl font-bold">{balance === null ? "—" : formatCurrency(balance)}</div>
          <div className="mt-2 text-xs text-gray-400">Updated live from the billing API.</div>
        </div>
      </section>

      <section className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-medium">Usage events</h3>
          {devActions && (
            <button
              onClick={onSimulate}
              disabled={simulating}
              className="inline-flex items-center px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {simulating ? "Simulating…" : "Simulate usage"}
            </button>
          )}
        </div>

        <div className="bg-white shadow rounded p-4">
          {loading ? (
            <div className="text-gray-500">Loading usage…</div>
          ) : error ? (
            <div className="text-red-600">Error: {error}</div>
          ) : !events || events.length === 0 ? (
            <div className="text-gray-600">No usage events yet. Try the simulate action (dev).</div>
          ) : (
            <ul role="list" className="divide-y">
              {events.map((ev) => (
                <li key={ev.id} className="py-3 flex justify-between items-start">
                  <div>
                    <div className="font-medium">{ev.description ?? "Usage event"}</div>
                    <div className="text-xs text-gray-500">{new Date(ev.createdAt).toLocaleString()}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm">{formatCurrency(ev.amountCents)}</div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>

      <footer className="text-xs text-gray-400">
        This page uses the local API endpoints at /api/v1/usage and /api/v1/billing. For production, set NEXT_PUBLIC_API_BASE_URL to your backend URL.
      </footer>
    </div>
  );
}
