/** @type {import('next').NextConfig} */
//
// The Python FastAPI backend runs on a different port (and inside Docker,
// a different host). We rewrite /api/* server-side so the browser only ever
// hits the frontend origin — no CORS dance.
//
// BACKEND_URL is read from the environment so the same image works both
// locally (defaults to http://localhost:8000) and in Docker Compose
// (set to http://backend:8000 via service name).
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig = {
  // Standalone output is required for the minimal production image
  // (next/dockerfile pattern — keeps the runtime container small).
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};
module.exports = nextConfig;
