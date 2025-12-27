import Link from 'next/link';

export default function DocsPage(): JSX.Element {
  return (
    <main className="max-w-4xl mx-auto px-4 py-12">
      <h1 className="text-3xl font-semibold mb-4">Docs</h1>

      <p className="text-gray-700 mb-6">
        Welcome to the documentation hub. Below are starter guides and repository-level resources to help
        you run, develop, and contribute to this project.
      </p>

      <section aria-labelledby="docs-overview" className="space-y-6">
        <h2 id="docs-overview" className="sr-only">
          Documentation overview
        </h2>

        <ul className="grid gap-4 sm:grid-cols-2">
          <li className="p-4 border rounded-lg bg-white/50">
            <Link href="/docs/getting-started" className="text-lg font-medium text-sky-600 hover:underline">
              Getting started
            </Link>
            <p className="text-sm text-gray-600 mt-1">
              Quickstart guide: how to run the app locally, run tests, and deploy. Recommended first read for new
              contributors.
            </p>
          </li>

          <li className="p-4 border rounded-lg bg-white/50">
            <a
              href="https://github.com/your-repo/clients/web#readme"
              target="_blank"
              rel="noopener noreferrer"
              className="text-lg font-medium text-sky-600 hover:underline"
            >
              Repository README
            </a>
            <p className="text-sm text-gray-600 mt-1">Repository-level documentation and contribution guidelines.</p>
          </li>
        </ul>
      </section>
    </main>
  );
}
