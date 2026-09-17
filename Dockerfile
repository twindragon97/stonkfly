FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install .

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/data /app/runs

VOLUME ["/app/data", "/app/runs"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
