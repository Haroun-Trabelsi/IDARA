# -----------------------------
# Video Difficulty Pipeline
# Production/Dev Dockerfile
# -----------------------------

FROM python:3.9-slim AS base

# Prevent python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS deps (if any heavy libs like opencv need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (leverages Docker layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source
COPY . .

# Default command – start watchdog-based pipeline
# Overridden by docker-compose which may run uvicorn
CMD ["python", "main.py"]
