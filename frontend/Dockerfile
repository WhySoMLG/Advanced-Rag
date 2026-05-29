# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
#  Frontend image — Next.js 14 production build (standalone output)
# ──────────────────────────────────────────────────────────────────────────────
# Three stages keep the runtime image tiny (~180 MB):
#   1. deps    — install node_modules (cached as long as package*.json unchanged)
#   2. builder — copy source + run `next build` (produces .next/standalone)
#   3. runner  — minimal node:alpine with only the standalone output
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. Dependency install ────────────────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ── 2. Build ─────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# IMPORTANT — Next.js standalone output does NOT ship next.config.js. The
# `rewrites()` block is evaluated during `next build` and the result is baked
# into the server bundle. That means BACKEND_URL has to be available HERE,
# not just at runtime, or the proxy will hard-code http://localhost:8000.
ARG BACKEND_URL=http://backend:8000
ENV BACKEND_URL=${BACKEND_URL}
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ── 3. Runtime ───────────────────────────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Run as a non-root user for hygiene.
RUN addgroup --system --gid 1001 nodejs \
 && adduser  --system --uid 1001 nextjs

# The standalone output already includes a minimal node_modules — that's the
# whole point. Public assets and static chunks are copied alongside.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
