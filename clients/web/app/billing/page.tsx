/* clients/web/app/billing/page.tsx

Simple Billing page for /app/billing
- Client-side page that fetches billing summary and transactions from the local API stubs.
- Renders a BillingCard and TransactionList and exposes Add funds + Manage billing flows.
*/

'use client'

import React, { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import BillingCard from '../../../components/BillingCard'
import TransactionList from '../../../components/TransactionList'
import { getBilling } from '../../lib/api'

// Types for API shapes used on this page
export interface Transaction {
  id: string
  amount: number
  currency?: string
  description?: string
  createdAt: string
  type?: 'credit' | 'debit'
}

export interface BillingSummary {
  balance: number
  currency?: string
  lastUpdated?: string
  transactions: Transaction[]
}

// Helper function to build API base (prefers NEXT_PUBLIC_API_BASE_URL but falls back to relative path)
function apiBase() {
  const env = typeof process !== 'undefined' ? (process.env.NEXT_PUBLIC_API_BASE_URL as string | undefined) : undefined
  return env && env !== '' ? env.replace(/\/+$/, '') : ''
}

export default function BillingPage(): JSX.Element {
  const [billing, setBilling] = useState<BillingSummary | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  async function load() {
    setLoading(true)
    setError(null)
    try {
      // Use centralized API helper to fetch billing; the helper returns values in cents.
      const resp = await getBilling()

      // Transform API response (balanceCents, transactions[].amountCents) into the UI-facing shape (major units)
      const transformed: BillingSummary = {
        balance: (resp.balanceCents ?? 0) / 100,
        currency: resp.currency,
        lastUpdated: resp.lastUpdated,
        transactions: (resp.transactions || []).map((t: any) => ({
          id: t.id,
          amount: (t.amountCents ?? 0) / 100,
          currency: t.currency,
          description: t.description,
          createdAt: t.createdAt,
          type: t.type,
        })),
      }

      setBilling(transformed)
    } catch (err: any) {
      console.error('billing load error', err)
      setError(err?.message ?? 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <main className="p-6 max-w-4xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Billing</h1>
        <p className="text-sm text-gray-600">Manage account balance, view recent transactions, and add funds.</p>
      </header>

      {loading && (
        <div className="space-y-4">
          <div className="h-24 bg-gray-100 animate-pulse rounded-md" />
          <div className="h-40 bg-gray-100 animate-pulse rounded-md" />
        </div>
      )}

      {!loading && error && (
        <div className="rounded-md p-4 bg-red-50 text-red-700">Error loading billing: {error}</div>
      )}

      {!loading && !error && billing && (
        <section className="space-y-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            {/* BillingCard is a small presentational component that shows balance + lastUpdated */}
            <BillingCard balance={billing.balance} lastUpdated={billing.lastUpdated} currency={billing.currency} />

            <div className="flex items-center gap-2">
              <AddFundsButton onSuccess={() => { load(); router.refresh() }} />
              <ManageBillingButton />
            </div>
          </div>

          <div>
            <h2 className="text-lg font-medium mb-2">Recent transactions</h2>
            <TransactionList transactions={billing.transactions} />
          </div>
        </section>
      )}

      {!loading && !error && !billing && (
        <div className="rounded-md p-6 bg-yellow-50 text-yellow-800">No billing information available.</div>
      )}

      <footer className="mt-8 text-sm text-gray-500">
        <p>Tip: In development, you can simulate a successful checkout using the dev API stubs.</p>
      </footer>
    </main>
  )
}

// --- Client-only controls ---
// These components run in the browser; they call the local Next.js API routes which can be stubbed.

function AddFundsButton({ onSuccess }: { onSuccess?: () => void }) {
  const [loading, setLoading] = useState(false)

  async function handleAddFunds() {
    setLoading(true)
    try {
      const base = apiBase() || ''
      const res = await fetch(`${base}/api/v1/checkout`, { method: 'POST' })
      if (!res.ok) throw new Error(`Checkout failed: ${res.status}`)
      const payload = await res.json()

      // If the stub or real API returns a stripe checkout url → redirect
      if (payload.url) {
        window.location.assign(payload.url)
        return
      }

      // Otherwise, treat as stubbed success; notify and refresh data via callback
      if (payload.success) {
        alert('Add funds: simulated success')
        onSuccess?.()
        return
      }

      // Fallback: maybe the API returned updated billing state directly
      if (payload.updated) {
        alert('Add funds: updated')
        onSuccess?.()
        return
      }

      throw new Error('Unexpected checkout response')
    } catch (err: any) {
      console.error('Add funds error', err)
      alert('Failed to start checkout: ' + (err?.message ?? 'unknown'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <button
      onClick={handleAddFunds}
      disabled={loading}
      className="inline-flex items-center px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-60"
    >
      {loading ? 'Processing…' : 'Add funds'}
    </button>
  )
}

function ManageBillingButton() {
  const [loading, setLoading] = useState(false)

  async function handleManage() {
    setLoading(true)
    try {
      const base = apiBase() || ''
      const res = await fetch(`${base}/api/v1/portal`, { method: 'POST' })
      if (!res.ok) throw new Error(`Portal failed: ${res.status}`)
      const payload = await res.json()

      if (payload.url) {
        // open Stripe customer portal or stubbed portal in a new tab
        window.open(payload.url, '_blank')
        return
      }

      // Stubbed fallback
      alert('Open billing portal (stubbed)')
    } catch (err: any) {
      console.error('Manage billing error', err)
      alert('Failed to open billing portal: ' + (err?.message ?? 'unknown'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <button
      onClick={handleManage}
      disabled={loading}
      className="inline-flex items-center px-3 py-2 border border-gray-200 rounded-md text-sm hover:bg-gray-50 disabled:opacity-60"
    >
      {loading ? 'Opening…' : 'Manage billing'}
    </button>
  )
}

