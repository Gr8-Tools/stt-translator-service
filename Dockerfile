# ── Build stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fail fast with a clear error if Python is too new for GigaAM deps.
RUN python3 -c "import sys; v=sys.version_info[:2]; ok=v < (3, 13);\nprint(f'Python {v[0]}.{v[1]} detected');\nraise SystemExit(0 if ok else 'Python >= 3.13 is not supported by GigaAM dependencies yet.')"

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

# ── Runtime ────────────────────────────────────────────────────────────────
EXPOSE 8000

# Default: 1 worker; adjust STT_WORKERS env var for multi-worker deployments.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
