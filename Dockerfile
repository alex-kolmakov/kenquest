# ── Stage 1: Build React frontend ──────────────────────────────────────────────
FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python runtime (FastAPI + embedded DBs) ───────────────────────────
FROM python:3.12-slim AS backend

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock* ./

# Install production dependencies (no dev extras)
RUN uv sync --frozen --no-dev

# Copy backend source
COPY backend/ ./backend/

# Embed frontend dist for nginx (copied out via named volume at compose startup)
COPY --from=frontend-build /app/frontend/dist /app/frontend_dist

# Create data directory (override with volume mount in production)
RUN mkdir -p /app/data/graph /app/data/chroma

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
