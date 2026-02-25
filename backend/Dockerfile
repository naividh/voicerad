# VoiceRad Backend Dockerfile
# Supports both CPU-only and CUDA GPU modes via build arg
#
# Build for CPU:  docker build -t voicerad-backend .
# Build for GPU:  docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 -t voicerad-backend-gpu .

ARG BASE_IMAGE=python:3.10-slim

FROM ${BASE_IMAGE}

WORKDIR /app

# Install system deps (Python if using CUDA base, plus audio/curl libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libsndfile1 \
        ffmpeg \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure 'python' and 'pip' commands work
    ln -sf /usr/bin/python3 /usr/bin/python || true && \
    ln -sf /usr/bin/pip3 /usr/bin/pip || true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "app.py"]
