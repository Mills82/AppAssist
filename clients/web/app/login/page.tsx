"use client";

// clients/web/app/login/page.tsx
// Client-side Login page for /app/login
// Collects email & password, validates, calls useAuth().signIn and redirects on success.

import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { useAuth } from "../../lib/useAuth";

export default function LoginPage(): JSX.Element {
  const router = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl = searchParams?.get("callbackUrl") || searchParams?.get("next") || "/app/dashboard";

  const { signIn, loading: authLoading } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState<{ email?: string; password?: string; form?: string }>({});
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    // Clear form error when fields change
    if (errors.form) setErrors((e) => ({ ...e, form: undefined }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [email, password]);

  const isValidEmail = (value: string) => /\S+@\S+\.\S+/.test(value);

  const validate = () => {
    const next: typeof errors = {};
    if (!email) next.email = "Email is required";
    else if (!isValidEmail(email)) next.email = "Enter a valid email address";

    if (!password) next.password = "Password is required";

    setErrors(next);
    return Object.keys(next).length === 0;
  };

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!validate()) return;
    setSubmitting(true);
    try {
      // signIn is expected to be provided by AuthProvider/useAuth and to either resolve on success or throw/reject on failure
      await signIn({ email, password });
      // Redirect to requested page (preserve callback) or default dashboard
      // Use replace so login doesn't remain in history
      router.replace(callbackUrl || "/app/dashboard");
    } catch (err: any) {
      // Normalize error message
      const message = (err && (err.message || err.error || String(err))) || "Login failed";
      setErrors({ form: message });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-md bg-white rounded-lg shadow p-6">
        <h1 className="text-2xl font-semibold mb-4">Sign in</h1>

        {errors.form && (
          <div role="alert" data-testid="error" className="mb-4 text-sm text-red-700 bg-red-50 p-2 rounded">
            {errors.form}
          </div>
        )}

        <form onSubmit={onSubmit} noValidate>
          <label className="block mb-2">
            <span className="text-sm font-medium text-gray-700">Email</span>
            <input
              data-testid="email"
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full rounded border-gray-300 shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="you@example.com"
              required
              aria-invalid={!!errors.email}
              aria-describedby={errors.email ? "email-error" : undefined}
            />
            {errors.email && (
              <p id="email-error" className="mt-1 text-xs text-red-600">
                {errors.email}
              </p>
            )}
          </label>

          <label className="block mb-4">
            <span className="text-sm font-medium text-gray-700">Password</span>
            <input
              data-testid="password"
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full rounded border-gray-300 shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="••••••••"
              required
              aria-invalid={!!errors.password}
              aria-describedby={errors.password ? "password-error" : undefined}
            />
            {errors.password && (
              <p id="password-error" className="mt-1 text-xs text-red-600">
                {errors.password}
              </p>
            )}
          </label>

          <button
            data-testid="submit"
            type="submit"
            disabled={submitting || authLoading}
            className={`w-full inline-flex items-center justify-center px-4 py-2 rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50`}
          >
            {submitting || authLoading ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <div className="mt-4 text-sm text-gray-600">
          Don’t have an account? <Link href="/app/signup" className="text-indigo-600 hover:underline">Create one</Link>
        </div>

        <div className="mt-4 text-xs text-gray-400">
          <p>Note: This page calls your frontend auth client (useAuth). In dev, the auth client may call a stubbed /api/v1 endpoint.</p>
        </div>
      </div>
    </main>
  );
}
