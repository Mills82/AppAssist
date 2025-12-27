'use client';
/* clients/web/context/AuthProvider.tsx
   AuthProvider: client-side session manager for AppAssist.
   - Persists a minimal session ({ token, user }) to sessionStorage under key 'app_session'.
   - Exposes signIn, signUp, signOut helpers and derived state via useAuth().
   - Uses dynamic import to call ../lib/api signIn/signUp when available; falls back to a dev stub otherwise.
*/

import React, { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";

// --------------------
// Types
// --------------------
export type AuthUser = {
  id: string;
  name?: string;
  email: string;
};

export type AuthSession = {
  token: string;
  user: AuthUser;
};

export type SignInCredentials = { email: string; password: string };
export type SignUpCredentials = { name?: string; email: string; password: string };

export type AuthContextValue = {
  user: AuthUser | null;
  token: string | null;
  loading: boolean;
  isAuthenticated: boolean;
  // Updated signatures: accept a single credentials object and return the session
  signIn: (creds: SignInCredentials) => Promise<AuthSession>;
  signUp: (creds: SignUpCredentials) => Promise<AuthSession>;
  // signOut returns a promise so callers can await cleanup/navigation
  signOut: () => Promise<void>;
};

// --------------------
// Constants
// --------------------
const SESSION_KEY = "app_session";

// --------------------
// Context
// --------------------
export const AuthContext = createContext<AuthContextValue | undefined>(undefined);

// --------------------
// Helpers / Fallbacks
// --------------------
async function devFallbackSignIn(email: string, _password: string): Promise<AuthSession> {
  // Small fake delay to simulate network
  await new Promise((r) => setTimeout(r, 350));
  const user: AuthUser = { id: `dev-${email}`, name: "Demo User", email };
  return { token: `dev-token-${Date.now()}`, user };
}

async function devFallbackSignUp(name: string | undefined, email: string, _password: string): Promise<AuthSession> {
  await new Promise((r) => setTimeout(r, 400));
  const user: AuthUser = { id: `dev-${email}`, name: name ?? "New User", email };
  return { token: `dev-token-${Date.now()}`, user };
}

function loadSessionFromStorage(): AuthSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AuthSession;
    if (parsed && parsed.token && parsed.user) return parsed;
    return null;
  } catch (err) {
    console.warn("AuthProvider: failed to parse session from sessionStorage", err);
    return null;
  }
}

function persistSessionToStorage(session: AuthSession | null) {
  if (typeof window === "undefined") return;
  try {
    if (!session) {
      sessionStorage.removeItem(SESSION_KEY);
    } else {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
    }
  } catch (err) {
    console.warn("AuthProvider: failed to persist session", err);
  }
}

// --------------------
// Provider (core, platform-agnostic)
// --------------------
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    // Restore session on mount (client only)
    const session = loadSessionFromStorage();
    if (session) {
      setUser(session.user);
      setToken(session.token);
    }
    setLoading(false);
  }, []);

  const signIn = async (creds: SignInCredentials): Promise<AuthSession> => {
    setLoading(true);
    try {
      const { email, password } = creds;
      // Try to dynamically import the api helper. This avoids hard compile-time dependency
      // so the provider can be added before ../lib/api.ts exists.
      const apiModule = typeof window !== "undefined" ? await import("../lib/api").catch(() => null) : null;
      let session: AuthSession;
      if (apiModule && typeof (apiModule as any).signIn === "function") {
        session = await (apiModule as any).signIn({ email, password } as SignInCredentials);
      } else {
        // Fallback dev stub
        session = await devFallbackSignIn(email, password);
      }

      setUser(session.user);
      setToken(session.token);
      persistSessionToStorage(session);
      return session;
    } finally {
      setLoading(false);
    }
  };

  const signUp = async (creds: SignUpCredentials): Promise<AuthSession> => {
    setLoading(true);
    try {
      const { name, email, password } = creds;
      const apiModule = typeof window !== "undefined" ? await import("../lib/api").catch(() => null) : null;
      let session: AuthSession;
      if (apiModule && typeof (apiModule as any).signUp === "function") {
        session = await (apiModule as any).signUp({ name, email, password } as SignUpCredentials);
      } else {
        session = await devFallbackSignUp(name, email, password);
      }

      setUser(session.user);
      setToken(session.token);
      persistSessionToStorage(session);
      return session;
    } finally {
      setLoading(false);
    }
  };

  const signOut = async (): Promise<void> => {
    // Return a promise so callers can await navigation/cleanup
    setUser(null);
    setToken(null);
    persistSessionToStorage(null);
    return Promise.resolve();
  };

  const value: AuthContextValue = {
    user,
    token,
    loading,
    isAuthenticated: !!token && !!user,
    signIn,
    signUp,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// --------------------
// Hook
// --------------------
export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within an AuthProvider. Wrap your app with <AuthProvider>.\nSee clients/web/context/AuthProvider.tsx");
  }
  return ctx;
}

// --------------------
// Client wrapper + route-guarding
// --------------------
// This component is the exported client-only provider wrapper that app/layout.tsx can import.
export function ClientAuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // Client-side redirect to login if trying to access /app/* while unauthenticated.
  useEffect(() => {
    // Only run in browser
    if (typeof window === "undefined") return;
    // If there's already a session in storage we don't redirect; AuthProvider will restore it.
    const session = loadSessionFromStorage();
    if (!session) {
      const path = pathname ?? window.location.pathname;
      if (path.startsWith("/app") && path !== "/app/login" && path !== "/app/signup") {
        const current = path + (window.location.search || "");
        const loginUrl = `/app/login?callbackUrl=${encodeURIComponent(current)}`;
        router.replace(loginUrl);
      }
    }
    // We intentionally do not include router in the deps beyond mount semantics.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  return (
    <AuthProvider>
      <PostSignInRedirectHandler>{children}</PostSignInRedirectHandler>
    </AuthProvider>
  );
}

// Internal component: when a session is created (token present) and there is a callbackUrl param,
// navigate to it and clear the param by replacing the history entry.
function PostSignInRedirectHandler({ children }: { children: ReactNode }) {
  const { token } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (typeof window === "undefined") return;
    const cb = searchParams?.get("callbackUrl");
    if (token && cb) {
      try {
        // Navigate to the callback URL and replace so the login page isn't in history.
        router.replace(cb);
      } catch (err) {
        // Fallback: push if replace fails
        router.push(cb);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  return <>{children}</>;
}

// --------------------
// Small usage demo component (can be imported into a page for quick manual testing)
// --------------------
export function AuthDemoCard() {
  const { user, token, loading, signIn, signUp, signOut, isAuthenticated } = useAuth();

  const doSignIn = async () => {
    try {
      await signIn({ email: "demo@example.com", password: "password" });
      // caller pages/components should show toasts; keep this demo minimal
    } catch (err) {
      console.error("Sign-in failed", err);
    }
  };

  const doSignUp = async () => {
    try {
      await signUp({ name: "Demo", email: "new@example.com", password: "password" });
    } catch (err) {
      console.error("Sign-up failed", err);
    }
  };

  return (
    <div style={{ border: "1px solid #e5e7eb", padding: 12, borderRadius: 8, maxWidth: 420 }}>
      <h3 style={{ marginTop: 0 }}>AuthDemo</h3>
      <div>Loading: {loading ? "yes" : "no"}</div>
      <div>Authenticated: {isAuthenticated ? "yes" : "no"}</div>
      <div>User: {user ? `${user.name ?? "(no-name)"} <${user.email}>` : "-"}</div>
      <div style={{ marginTop: 8 }}>
        <button onClick={doSignIn} style={{ marginRight: 8 }}>
          Sign In (demo)
        </button>
        <button onClick={doSignUp} style={{ marginRight: 8 }}>
          Sign Up (demo)
        </button>
        <button onClick={() => { signOut().catch(console.error); }}>Sign Out</button>
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: "#6b7280" }}>Token: {token ?? "-"}</div>
    </div>
  );
}

// Note: The demo component expects to be rendered inside an <AuthProvider> or ClientAuthProvider. Example:
//
// import { ClientAuthProvider, AuthDemoCard } from "clients/web/context/AuthProvider";
//
// function App() {
//   return (
//     <ClientAuthProvider>
//       <AuthDemoCard />
//     </ClientAuthProvider>
//   );
// }

export default ClientAuthProvider;
