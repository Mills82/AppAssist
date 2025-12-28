'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function ClientHeader() {
  const pathname = usePathname() || '/';

  const navLinkProps = (path: string): { className: string; 'aria-current'?: 'page' } => {
    const active = path === '/' ? pathname === '/' : pathname.startsWith(path);
    return {
      className: `${active ? 'text-sky-600 font-semibold' : 'text-slate-700'} hover:underline px-2 py-1 rounded`,
      'aria-current': active ? 'page' : undefined,
    };
  };

  const dashboardActive = pathname.startsWith('/app');

  return (
    <header className="bg-white shadow">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-x-4">
          <Link href="/" className="text-lg font-semibold text-slate-900">
            AppAssist
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
            className={`text-sm px-3 py-1 rounded ${
              dashboardActive ? 'bg-sky-700 text-white' : 'bg-sky-600 text-white hover:bg-sky-700'
            }`}
            aria-current={dashboardActive ? 'page' : undefined}
          >
            Dashboard
          </Link>
        </div>
      </div>
    </header>
  );
}
