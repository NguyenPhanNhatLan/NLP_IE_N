# syntax=docker/dockerfile:1.4
FROM python:3.10.19-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Cài đặt build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy và build dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.10.19-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf/transformers \
    PATH="/home/app/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system --gid 1001 app \
    && adduser --system --uid 1001 --ingroup app app \
    && mkdir -p /opt/hf /app \
    && chown -R app:app /opt/hf /app


COPY --from=builder /wheels /wheels

RUN pip install --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]