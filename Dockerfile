# ── Build stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y \
        python3-pip \
        python3-venv \
        ffmpeg

RUN python3 --version

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python3 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache-bust only the source layer when needed.
ARG CODE_CACHEBUST
RUN echo "Code cache bust: $CODE_CACHEBUST"

# Copy application source
COPY app/ ./app/

# ── Runtime ────────────────────────────────────────────────────────────────
EXPOSE 8000

# Default: 1 worker; adjust STT_WORKERS env var for multi-worker deployments.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
