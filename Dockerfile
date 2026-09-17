FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STONKFLY_DATA=/app/data \
    STONKFLY_RUN=/app/runs/paper \
    STONKFLY_VIZ_CACHE=/app/dashboard/cache \
    HOST=0.0.0.0 \
    PORT=8765

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE THIRD_PARTY.md ./

COPY stonkfly ./stonkfly
COPY dashboard ./dashboard

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install .

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/data \
    && mkdir -p /app/runs/paper \
    && mkdir -p /app/dashboard/cache

EXPOSE 8765

VOLUME ["/app/data", "/app/runs", "/app/dashboard/cache"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
