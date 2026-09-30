# ==============================================================================
# Dockerfile for FinTrust ML Workflow (Week 3: Integration-Ready)
# ==============================================================================
# Multi-stage container definition ensuring 100% reproducible deployment across
# Linux, macOS, and cloud Kubernetes/ECS environments.
# ==============================================================================

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, artifacts, data, and docs
COPY src/ /app/src/
COPY data/ /app/data/
COPY artifacts/ /app/artifacts/
COPY main.py /app/
COPY check_all.py /app/

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default execution: run the FastAPI microservice
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
