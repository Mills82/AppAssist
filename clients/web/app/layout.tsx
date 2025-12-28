/* clients/web/app/layout.tsx
 * Root layout for the Next.js App Router.
 * Provides global styles and top navigation for public pages and /app area.
 */

import './globals.css';
import dynamic from 'next/dynamic';
import type { ReactNode } from 'react';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'AppAssist',
  description: 'AppAssist — Productivity tools and integrations for apps (AppAssist.ai)',
};

const Header = dynamic(() => import('../components/ClientHeader'), {
  ssr: false,
  loading: () => (
    <header className="bg-white shadow">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-x-4">
          <div className="text-lg font-semibold text-slate-900">AppAssist</div>
          <nav className="hidden md:flex items-center gap-x-3 text-sm text-slate-700">
            <span className="px-2 py-1 rounded">Home</span>
            <span className="px-2 py-1 rounded">Pricing</span>
            <span className="px-2 py-1 rounded">Docs</span>
            <span className="px-2 py-1 rounded">About</span>
            <span className="px-2 py-1 rounded">Contact</span>
          </nav>
        </div>

        <div className="flex items-center gap-x-3">
          <div className="bg-sky-600 text-white text-sm px-3 py-1 rounded">Dashboard</div>
        </div>
      </div>
    </header>
  ),
});

// Client-side AuthProvider + guard wrapper. Loaded dynamically with ssr: false so this module remains a server component.
const ClientAuthProvider = dynamic(
  () => import('../context/AuthProvider').then((m) => (m.ClientAuthProvider || m.AuthProvider || m.default)),
  { ssr: false, loading: () => <div /> }
);

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 text-gray-900">
        <Header />

        <ClientAuthProvider>
          <main className="container mx-auto p-6">{children}</main>
        </ClientAuthProvider>

        <footer className="border-t bg-white">
          <div className="container mx-auto px-4 py-6 text-sm text-gray-600">
            © {new Date().getFullYear()} AppAssist.ai — Built with Next.js & Tailwind
          </div>
        </footer>
      </body>
    </html>
  );
}
