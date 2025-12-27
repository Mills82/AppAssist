import React from 'react';

export default function PricingPage(): JSX.Element {
  return (
    <main className="container mx-auto px-4 py-12">
      <header className="max-w-3xl mx-auto text-center mb-10">
        <h1 className="text-3xl md:text-4xl font-semibold text-gray-900 dark:text-gray-100">Pricing</h1>
        <p className="mt-4 text-gray-600 dark:text-gray-300">Choose a plan that suits your needs. No credit card required to start with Free.</p>
      </header>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
        <article className="border rounded-lg p-6 shadow-sm bg-white dark:bg-gray-800">
          <h2 id="plan-free" className="text-xl font-semibold text-gray-900 dark:text-gray-100">Free</h2>
          <p className="mt-2 text-3xl font-bold text-gray-900 dark:text-gray-100">$0<span className="text-base font-medium text-gray-500">/mo</span></p>

          <ul className="mt-4 space-y-2 text-gray-600 dark:text-gray-300">
            <li>Basic features for personal use</li>
            <li>Limited to 1 user and 3 projects</li>
            <li>Email support</li>
          </ul>

          <div className="mt-6">
            <a href="/signup" className="inline-block w-full text-center px-4 py-2 bg-gray-100 text-gray-900 rounded-md hover:bg-gray-200">Get started</a>
          </div>
        </article>

        <article className="border-2 border-indigo-600 rounded-lg p-6 shadow-md bg-white dark:bg-gray-800">
          <h2 id="plan-pro" className="text-xl font-semibold text-gray-900 dark:text-gray-100">Pro</h2>
          <p className="mt-2 text-3xl font-bold text-indigo-600">$12<span className="text-base font-medium text-indigo-500">/mo</span></p>

          <ul className="mt-4 space-y-2 text-gray-600 dark:text-gray-300">
            <li>All Free features plus advanced integrations</li>
            <li>Unlimited projects and priority support</li>
            <li>Team management and SSO</li>
          </ul>

          <div className="mt-6">
            <a href="/signup?plan=pro" className="inline-block w-full text-center px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">Choose Pro</a>
          </div>
        </article>
      </section>

      <section className="max-w-3xl mx-auto mt-12 text-center text-sm text-gray-500 dark:text-gray-400">
        <p>Questions about pricing or volume discounts? <a href="/contact" className="text-indigo-600 hover:underline">Contact sales</a>.</p>
      </section>
    </main>
  );
}
