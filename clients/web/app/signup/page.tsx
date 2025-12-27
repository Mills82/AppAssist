/* clients/web/app/signup/page.tsx */
"use client";

import React, { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useAuth } from "../../lib/useAuth";

// Simple email regex for client-side validation
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

type SignUpFormState = {
  name: string;
  email: string;
  password: string;
};

export default function SignUpPage() {
  const auth = useAuth();
  const router = useRouter();
  const params = useSearchParams();
  const callbackUrl = params?.get("callbackUrl") || "/app/dashboard";

  const [form, setForm] = useState<SignUpFormState>({ name: "", email: "", password: "" });
  const [errors, setErrors] = useState<Partial<Record<keyof SignUpFormState, string>>>({});
  const [loading, setLoading] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  function validate(): boolean {
    const next: Partial<Record<keyof SignUpFormState, string>> = {};
    if (!form.name.trim()) next.name = "Name is required";
    if (!form.email.trim()) next.email = "Email is required";
    else if (!EMAIL_RE.test(form.email)) next.email = "Enter a valid email";
    if (!form.password) next.password = "Password is required";
    setErrors(next);
    return Object.keys(next).length === 0;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    if (!validate()) return;
    if (!auth || typeof auth.signUp !== "function") {
      setSubmitError("Auth not available. Please implement useAuth.signUp().");
      return;
    }

    try {
      setLoading(true);
      // signUp is expected to resolve to { token, user } or throw an Error
      await auth.signUp({ name: form.name.trim(), email: form.email.trim(), password: form.password });
      // on success redirect to intended route
      router.push(callbackUrl);
    } catch (err: any) {
      console.error("Sign up failed", err);
      setSubmitError(err?.message || "Sign up failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-md bg-white rounded-lg shadow px-6 py-8">
        <h1 className="text-2xl font-semibold mb-2">Create an account</h1>
        <p className="text-sm text-gray-600 mb-6">Start your AppAssist trial — no credit card required.</p>

        <form onSubmit={handleSubmit} noValidate>
          <label className="block mb-3">
            <span className="text-sm font-medium">Full name</span>
            <input
              aria-label="Full name"
              value={form.name}
              onChange={(e) => setForm((s) => ({ ...s, name: e.target.value }))}
              className="mt-1 block w-full rounded-md border-gray-200 shadow-sm focus:ring-2 focus:ring-indigo-300 px-3 py-2"
              placeholder="Jane Doe"
              required
              disabled={loading}
            />
            {errors.name && <div className="text-xs text-red-600 mt-1">{errors.name}</div>}
          </label>

          <label className="block mb-3">
            <span className="text-sm font-medium">Email</span>
            <input
              aria-label="Email"
              type="email"
              value={form.email}
              onChange={(e) => setForm((s) => ({ ...s, email: e.target.value }))}
              className="mt-1 block w-full rounded-md border-gray-200 shadow-sm focus:ring-2 focus:ring-indigo-300 px-3 py-2"
              placeholder="you@example.com"
              required
              disabled={loading}
            />
            {errors.email && <div className="text-xs text-red-600 mt-1">{errors.email}</div>}
          </label>

          <label className="block mb-4">
            <span className="text-sm font-medium">Password</span>
            <input
              aria-label="Password"
              type="password"
              value={form.password}
              onChange={(e) => setForm((s) => ({ ...s, password: e.target.value }))}
              className="mt-1 block w-full rounded-md border-gray-200 shadow-sm focus:ring-2 focus:ring-indigo-300 px-3 py-2"
              placeholder="Choose a secure password"
              required
              disabled={loading}
            />
            {errors.password && <div className="text-xs text-red-600 mt-1">{errors.password}</div>}
          </label>

          <div className="flex items-center justify-between">
            <button
              type="submit"
              className="inline-flex items-center justify-center px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-60"
              disabled={loading}
            >
              {loading ? "Creating account..." : "Create account"}
            </button>
            <Link href="/app/login" className="text-sm text-gray-600 hover:underline">
              Already have an account?
            </Link>
          </div>

          {submitError && <div className="mt-4 text-sm text-red-600">{submitError}</div>}
        </form>

        <div className="mt-6 text-xs text-gray-500">
          By creating an account you agree to our <a className="underline">Terms</a>.
        </div>
      </div>
    </div>
  );
}

// Minimal development-time smoke check so importing this module logs a tiny message in Node (useful for quick testing)
if (typeof window === "undefined") {
  // This runs when the module is loaded in a Node environment (e.g., static analyzers, tests)
  // Keep this side-effect tiny and safe.
  // eslint-disable-next-line no-console
  console.log("[clients/web/app/signup/page] module loaded");
}
