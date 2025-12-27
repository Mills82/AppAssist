/* clients/web/lib/useAuth.ts
 * Lightweight typed hook that exposes the AuthProvider context to pages/components.
 * This hook must be used from client components (they should include the 'use client' directive).
 */

import React, { useContext } from "react";
import { AuthContext, AuthContextValue } from "../context/AuthProvider";

/** Minimal user shape used by the auth context */
export interface User {
  id: string;
  name?: string | null;
  email: string;
}

/** Credentials expected by signIn / signUp helpers */
export interface AuthCredentials {
  email: string;
  password: string;
  name?: string;
}

/**
 * Re-export the provider's context value shape so callers can rely on a single source of truth.
 * This preserves the previous `AuthHookReturn` identifier for compatibility while ensuring
 * the type is the same as exported from the AuthProvider implementation.
 */
export type AuthHookReturn = AuthContextValue;

/**
 * useAuth
 * A thin consumer hook that reads the AuthContext and re-exports the typed API.
 * It intentionally performs no I/O or persistence itself — the AuthProvider implements that.
 *
 * Note: this module does NOT include a 'use client' directive. Callers (React components)
 * that use this hook must be client components and include 'use client' at the top of their file.
 *
 * Throws a helpful error when used outside of an AuthProvider to avoid silent failures.
 */
export function useAuth(): AuthHookReturn {
  // Use the AuthContext exported by the provider and assert the expected shape.
  const ctx = useContext(AuthContext as React.Context<AuthHookReturn | null>);

  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider (wrap your app with AuthProvider)");
  }

  return ctx;
}

// Also provide a default export for consumers that import the hook as default.
export default useAuth;

/*
Example (usage in a client component):

// "use client"
// import React from 'react'
// import useAuth from 'lib/useAuth'

// export default function ProfileButton() {
//   const { user, signOut, loading } = useAuth();
//   if (loading) return <button disabled>Loading...</button>;
//   if (!user) return <a href="/app/login">Sign in</a>;
//   return (
//     <div>
//       <span>{user.name ?? user.email}</span>
//       <button onClick={() => signOut()}>Sign out</button>
//     </div>
//   );
// }
*/
