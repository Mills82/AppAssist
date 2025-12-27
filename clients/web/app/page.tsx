// clients/web/app/page.tsx
// Home (server) page for the Next.js App Router scaffold.

import Link from 'next/link'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'AI Dev Bot — Home',
  description: 'Inspect, analyze, and suggest safe edits to your codebase — built for developer workflows.',
}

export default function Home(): JSX.Element {
  return (
    <main className="min-h-screen flex flex-col items-center justify-start py-16 px-6 bg-white text-slate-900">
      <header className="w-full max-w-4xl">
        <nav className="flex items-center justify-between py-4">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-semibold">AI Dev Bot</h2>
            <span className="text-sm text-slate-500">Developer-first code assistance</span>
          </div>
          <div className="flex gap-4 text-sm text-slate-600">
            <Link href="/" className="hover:underline">Home</Link>
            <Link href="/pricing" className="hover:underline">Pricing</Link>
            <Link href="/docs" className="hover:underline">Docs</Link>
            <Link href="/about" className="hover:underline">About</Link>
            <Link href="/contact" className="hover:underline">Contact</Link>
          </div>
        </nav>
      </header>

      <section className="w-full max-w-4xl text-center mt-12">
        <h1 className="text-4xl sm:text-5xl font-bold">AI Dev Bot</h1>
        <p className="mt-4 text-lg text-slate-700">
          Inspect, analyze, and suggest safe edits to your codebase — built for developer workflows.
        </p>
        <div className="mt-8 flex justify-center gap-4">
          <Link href="/app/dashboard" className="inline-block bg-blue-600 hover:bg-blue-700 transition-colors text-white px-5 py-3 rounded-md">
            Open Dashboard
          </Link>
          <Link href="/docs" className="inline-block border border-slate-200 hover:border-slate-300 text-slate-700 px-4 py-3 rounded-md">
            Read Docs
          </Link>
        </div>

        <div className="mt-12 text-sm text-slate-600">
          <p>
            This scaffold uses Next.js App Router + TypeScript + Tailwind. Developers can visit /app/dashboard to validate the
            authenticated dashboard area and the chat-first layout.
          </p>
        </div>
      </section>

      <footer className="w-full max-w-4xl mt-16 py-6 border-t border-slate-100 text-sm text-slate-500">
        <div className="flex justify-between">
          <div>© {new Date().getFullYear()} AI Dev Bot</div>
          <div className="flex gap-4">
            <Link href="/pricing" className="hover:underline">Pricing</Link>
            <Link href="/docs" className="hover:underline">Docs</Link>
          </div>
        </div>
      </footer>
    </main>
  )
}
