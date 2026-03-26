FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first for better caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/
COPY data/ data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uv", "run", "uvicorn", "rag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
