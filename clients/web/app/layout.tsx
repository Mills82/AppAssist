"use client";
/* clients/web/app/layout.tsx
 * Root layout for the Next.js App Router.
 * Provides global styles and top navigation for public pages and /app area.
 */

import './globals.css';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import type { ReactNode } from 'react';
import { usePathname } from 'next/navigation';

// Client-side header implemented via next/dynamic so RootLayout stays a server component.
const Header = dynamic(
  () =>
    Promise.resolve(function ClientHeader() {
      'use client';
      const pathname = usePathname() || '/';

      const navLinkProps = (path: string) => {
        const active = path === '/' ? pathname === '/' : pathname.startsWith(path);
        return {
          className: `${active ? 'text-sky-600 font-semibold' : 'text-slate-700'} hover:underline px-2 py-1 rounded`,
          'aria-current': active ? 'page' : undefined,
        } as any;
      };

      const dashboardActive = pathname.startsWith('/app');

      return (
        <header className="bg-white shadow">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-x-4">
              <Link href="/" className="text-lg font-semibold text-slate-900">
                AI Dev Bot
              </Link>
              <nav className="hidden md:flex items-center gap-x-3 text-sm">
                <Link href="/" {...navLinkProps('/')}>Home</Link>
                <Link href="/pricing" {...navLinkProps('/pricing')}>Pricing</Link>
                <Link href="/docs" {...navLinkProps('/docs')}>Docs</Link>
                <Link href="/about" {...navLinkProps('/about')}>About</Link>
                <Link href="/contact" {...navLinkProps('/contact')}>Contact</Link>
              </nav>
            </div>

            <div className="flex items-center gap-x-3">
              <Link
                href="/app/dashboard"
                className={`text-sm px-3 py-1 rounded ${dashboardActive ? 'bg-sky-700 text-white' : 'bg-sky-600 text-white hover:bg-sky-700'}`}
                aria-current={dashboardActive ? 'page' : undefined}
              >
                Dashboard
              </Link>
            </div>
          </div>
        </header>
      );
    }),
  {
    ssr: false,
    // Loading fallback: render the original server-side header markup to avoid layout shift.
    loading: () => (
      <header className="bg-white shadow">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-x-4">
            <Link href="/" className="text-lg font-semibold text-slate-900">
              AI Dev Bot
            </Link>
            <nav className="hidden md:flex items-center gap-x-3 text-sm text-slate-700">
              <Link href="/" className="hover:underline px-2 py-1 rounded">Home</Link>
              <Link href="/pricing" className="hover:underline px-2 py-1 rounded">Pricing</Link>
              <Link href="/docs" className="hover:underline px-2 py-1 rounded">Docs</Link>
              <Link href="/about" className="hover:underline px-2 py-1 rounded">About</Link>
              <Link href="/contact" className="hover:underline px-2 py-1 rounded">Contact</Link>
            </nav>
          </div>

          <div className="flex items-center gap-x-3">
            <Link href="/app/dashboard" className="bg-sky-600 text-white text-sm px-3 py-1 rounded hover:bg-sky-700">Dashboard</Link>
          </div>
        </div>
      </header>
    ),
  }
);

// Client-side AuthProvider + guard wrapper. Loaded dynamically with ssr: false so this module remains a server component.
// NOTE: the imported module may export the provider under different names depending on implementation
// (ClientAuthProvider, AuthProvider, or default). We prefer ClientAuthProvider but fall back to other exports
// to be robust. The actual redirect/guarding logic runs client-side inside the provider so server components
// can still render without blocking on auth.
const ClientAuthProvider = dynamic(
  () => import('../context/AuthProvider').then((m) => (m.ClientAuthProvider || m.AuthProvider || m.default)),
  {
    ssr: false,
    // minimal loading fallback to avoid layout shift; the provider will handle redirects on the client
    loading: () => <div />,
  }
);

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>AI Dev Bot</title>
      </head>
      <body className="min-h-screen bg-gray-50 text-gray-900">
        <Header />

        {/* ClientAuthProvider runs only on the client (ssr: false). It should read sessionStorage, provide auth context, and redirect unauthenticated /app/* routes to /app/login while exempting /app/login and /app/signup. */}
        <ClientAuthProvider>
          <main className="container mx-auto p-6">
            {children}
          </main>
        </ClientAuthProvider>

        <footer className="border-t bg-white">
          <div className="container mx-auto px-4 py-6 text-sm text-gray-600">
            © {new Date().getFullYear()} AI Dev Bot — Built with Next.js & Tailwind
          </div>
        </footer>
      </body>
    </html>
  );
}
