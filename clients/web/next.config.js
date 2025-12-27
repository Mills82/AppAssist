// clients/web/next.config.js
// Next.js configuration: development-only proxy rewrites for /api/v1/* -> http://localhost:8000/api/v1/*

// This file enables a simple dev-only proxy so the frontend can call /api/v1/*
// without needing CORS or changing request URLs during local development.
const isDev = process.env.NODE_ENV !== 'production';

module.exports = {
  reactStrictMode: true,

  // Add development-only rewrites to forward API calls to the local Python backend.
  // The rule preserves path segments using :path* on both source and destination.
  async rewrites() {
    if (isDev) {
      return [
        {
          source: '/api/v1/:path*',
          destination: 'http://localhost:8000/api/v1/:path*',
        },
      ];
    }

    // No rewrites in production — the app should call the real API host in prod builds.
    return [];
  },
};

// Minimal runtime check: run `node clients/web/next.config.js` to inspect rewrites.
// This does not start Next.js; it's only a convenience to verify the dev proxy rule.
if (require.main === module) {
  (async () => {
    console.log('clients/web/next.config.js — runtime check');
    console.log('NODE_ENV=' + process.env.NODE_ENV);
    const cfg = module.exports;
    try {
      const r = await cfg.rewrites();
      console.log('rewrites:', JSON.stringify(r, null, 2));
    } catch (err) {
      console.error('error calling rewrites():', err);
      process.exitCode = 1;
    }
  })();
}
