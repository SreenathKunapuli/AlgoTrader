FROM node:20-alpine AS deps
WORKDIR /app
COPY web/package.json web/package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY web .
COPY --from=deps /app/node_modules ./node_modules
# API_INTERNAL_BASE is a build-time var for SSR routes that call FastAPI directly
ARG API_INTERNAL_BASE=http://api:8000
ENV API_INTERNAL_BASE=${API_INTERNAL_BASE}
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
# standalone output bundles everything; public and static dirs still need copying
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

ENV PORT=3000
CMD ["node", "server.js"]
