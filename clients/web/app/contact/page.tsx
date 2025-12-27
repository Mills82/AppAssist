'use client';

import React, { useState } from 'react';

export default function ContactPage(): JSX.Element {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ type: 'idle' | 'success' | 'error'; message?: string }>({ type: 'idle' });

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setStatus({ type: 'idle' });

    // Basic client-side validation (required attributes on inputs are primary)
    if (!name.trim() || !email.trim() || !message.trim()) {
      setStatus({ type: 'error', message: 'Please complete all required fields.' });
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('/api/v1/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, message }),
      });

      if (res.ok) {
        setStatus({ type: 'success', message: 'Thanks — your message was sent. We will follow up shortly.' });
        setName('');
        setEmail('');
        setMessage('');
      } else {
        // Try to read an error message from the response body, fallback to generic text
        let errorText = 'Something went wrong. Please try again later.';
        try {
          const payload = await res.json();
          if (payload && payload.message) errorText = String(payload.message);
        } catch {
          // ignore JSON parse errors
        }
        setStatus({ type: 'error', message: errorText });
      }
    } catch (err) {
      setStatus({ type: 'error', message: 'Network error. Please check your connection and try again.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="max-w-3xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-semibold mb-4">Contact Us</h1>
      <p className="text-gray-700 mb-6">Have questions or feedback? Fill out the form below and we'll get back to you.</p>

      <form onSubmit={handleSubmit} className="bg-white shadow-sm rounded-md p-6" aria-describedby="contact-form-desc">
        <div id="contact-form-desc" className="sr-only">
          Contact form with name, email, and message fields. Required fields are marked.
        </div>

        <div className="grid grid-cols-1 gap-4">
          <div>
            <label htmlFor="contact-name" className="block text-sm font-medium text-gray-700">
              Name
              <span aria-hidden className="text-red-500"> *</span>
            </label>
            <input
              id="contact-name"
              name="name"
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            />
          </div>

          <div>
            <label htmlFor="contact-email" className="block text-sm font-medium text-gray-700">
              Email
              <span aria-hidden className="text-red-500"> *</span>
            </label>
            <input
              id="contact-email"
              name="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            />
          </div>

          <div>
            <label htmlFor="contact-message" className="block text-sm font-medium text-gray-700">
              Message
              <span aria-hidden className="text-red-500"> *</span>
            </label>
            <textarea
              id="contact-message"
              name="message"
              required
              rows={6}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-md hover:bg-indigo-700 disabled:opacity-50"
            >
              {loading ? 'Sending…' : 'Send message'}
            </button>
          </div>

          <div role="status" aria-live="polite" className="pt-2">
            {status.type === 'success' && (
              <p className="text-sm text-green-700 bg-green-50 rounded-md p-2">{status.message}</p>
            )}
            {status.type === 'error' && (
              <p className="text-sm text-red-700 bg-red-50 rounded-md p-2">{status.message}</p>
            )}
          </div>
        </div>
      </form>

      <section className="mt-8 text-sm text-gray-600">
        <p>
          This form posts to <code className="bg-gray-100 px-1 rounded">/api/v1/contact</code>. In development the backend may return a mock
          response — this page will display success or error messages based on the response.
        </p>
      </section>
    </main>
  );
}
