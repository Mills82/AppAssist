import Link from 'next/link';
import React from 'react';

export default function AboutPage(): JSX.Element {
  return (
    <main className="max-w-3xl mx-auto px-4 py-12">
      <article>
        <h1 className="text-3xl font-bold mb-4">About</h1>

        <p className="mb-4 text-base leading-relaxed">
          This project is a small, focused web application that demonstrates the
          team's conventions for building with Next.js, React and Tailwind CSS.
          It provides a starting point for demos, documentation, and iterative
          feature development while keeping pages fast and accessible by
          rendering them on the server.
        </p>

        <p className="mb-4 text-base leading-relaxed">
          Our goal is to make the developer experience predictable and the user
          experience clear. You can find implementation notes and usage details
          in the <Link href="/docs" className="text-indigo-600 hover:underline">Docs</Link>, or
          check the <Link href="/dashboard" className="text-indigo-600 hover:underline">Dashboard</Link> to see
          example data and integrations.
        </p>

        <p className="mt-6 text-sm text-gray-600">
          If you need to contact the team, visit the <Link href="/contact" className="text-indigo-600 hover:underline">Contact</Link> page.
        </p>
      </article>
    </main>
  );
}
