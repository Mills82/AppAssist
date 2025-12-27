/* clients/web/components/TransactionList.tsx
   Simple Tailwind-styled TransactionList React component.
   Renders loading skeletons, an error banner with retry, an empty state with CTA,
   or a list of transactions. Exported types are included for local use/tests.
*/

"use client";

import React from "react";

export type Transaction = {
  id: string;
  // Monetary amount. Consumer can choose cents or decimal dollars; component
  // displays the number using Intl.NumberFormat. Prefer dollars (e.g. 12.34).
  amount: number;
  currency?: string; // ISO currency code, e.g. 'USD'
  description?: string;
  createdAt: string; // ISO timestamp
  status?: "pending" | "succeeded" | "failed";
};

export interface Props {
  transactions?: Transaction[];
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  // Optional CTA when empty
  emptyCtaLabel?: string;
  onEmptyCta?: () => void;
}

function formatAmount(amount: number, currency = "USD") {
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
    }).format(amount);
  } catch (e) {
    // Fallback
    return `${currency} ${amount}`;
  }
}

function statusClass(status?: Transaction["status"]) {
  switch (status) {
    case "succeeded":
      return "bg-green-100 text-green-800";
    case "pending":
      return "bg-yellow-100 text-yellow-800";
    case "failed":
      return "bg-red-100 text-red-800";
    default:
      return "bg-gray-100 text-gray-800";
  }
}

export default function TransactionList({
  transactions = [],
  loading = false,
  error = null,
  onRetry,
  emptyCtaLabel = "Add funds",
  onEmptyCta,
}: Props) {
  // Loading state: show a few skeleton rows
  if (loading) {
    return (
      <div className="w-full">
        <ul className="divide-y divide-gray-100">
          {[1, 2, 3].map((n) => (
            <li
              key={n}
              data-testid="tx-skeleton"
              className="flex items-center justify-between py-4 animate-pulse"
            >
              <div className="space-y-2">
                <div className="h-4 w-40 bg-gray-200 rounded" />
                <div className="h-3 w-32 bg-gray-200 rounded mt-1" />
              </div>
              <div className="h-4 w-20 bg-gray-200 rounded" />
            </li>
          ))}
        </ul>
      </div>
    );
  }

  // Error state: show banner with retry
  if (error) {
    return (
      <div className="w-full">
        <div
          className="rounded-md bg-red-50 p-4 mb-4"
          role="alert"
          aria-live="polite"
        >
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg
                className="h-5 w-5 text-red-400"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                />
              </svg>
            </div>
            <div className="ml-3 w-0 flex-1">
              <p className="text-sm font-medium text-red-800">Unable to load transactions</p>
              <p className="mt-1 text-sm text-red-700">{error}</p>
            </div>
            <div className="ml-4 flex-shrink-0">
              <button
                data-testid="tx-retry"
                onClick={onRetry}
                className="inline-flex items-center px-3 py-1.5 border border-transparent text-sm leading-5 font-medium rounded-md text-red-700 bg-red-100 hover:bg-red-200"
                aria-label="Retry loading transactions"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Empty state
  if (!transactions || transactions.length === 0) {
    return (
      <div className="w-full text-center py-10" data-testid="tx-empty">
        <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-gray-100">
          <svg
            className="h-6 w-6 text-gray-500"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h4v11H3zM10 3h11v18H10z" />
          </svg>
        </div>
        <h3 className="mt-4 text-sm font-medium text-gray-900">No transactions yet</h3>
        <p className="mt-2 text-sm text-gray-500">Once you add funds or use the service, transactions will appear here.</p>
        <div className="mt-4">
          <button
            onClick={onEmptyCta}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700"
          >
            {emptyCtaLabel}
          </button>
        </div>
      </div>
    );
  }

  // Normal list rendering
  return (
    <div className="w-full">
      <ul className="divide-y divide-gray-100">
        {transactions.map((t) => (
          <li
            key={t.id}
            data-testid={`tx-item-${t.id}`}
            className="flex items-center justify-between py-4"
          >
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="h-10 w-10 rounded-full bg-gray-50 flex items-center justify-center">
                  <svg
                    className="h-5 w-5 text-gray-400"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    aria-hidden="true"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2-1.343-2-3-2z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14v6" />
                  </svg>
                </div>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900">{t.description ?? "Transaction"}</p>
                <p className="mt-1 text-xs text-gray-500">{new Date(t.createdAt).toLocaleString()}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-sm font-medium text-gray-900">{formatAmount(t.amount, t.currency)}</div>
              <span
                className={`${statusClass(t.status)} inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium`}
                aria-hidden="true"
              >
                {t.status ?? "unknown"}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Small example/demo export that can be imported into a story or dev page.
export const Example = () => {
  const demo: Transaction[] = [
    { id: "tx_1", amount: 29.99, currency: "USD", description: "Add funds - card", createdAt: new Date().toISOString(), status: "succeeded" },
    { id: "tx_2", amount: -3.5, currency: "USD", description: "Usage - inference", createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), status: "succeeded" },
    { id: "tx_3", amount: 10, currency: "USD", description: "Add funds - test", createdAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(), status: "pending" },
  ];

  return (
    <div className="max-w-lg mx-auto p-4">
      <h4 className="text-lg font-semibold mb-3">TransactionList Demo</h4>
      <TransactionList transactions={demo} />
    </div>
  );
};

export type { Props };
