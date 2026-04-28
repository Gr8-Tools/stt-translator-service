# ── Build stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Store Hugging Face model weights in a predictable location
    # (mapped to a named Docker volume in docker-compose.yml)
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

# ── Runtime ────────────────────────────────────────────────────────────────
EXPOSE 8000

# Default: 1 worker; adjust STT_WORKERS env var for multi-worker deployments.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
