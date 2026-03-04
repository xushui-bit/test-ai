FROM python:3.11-slim

LABEL maintainer="your-email@example.com"
LABEL description="BGE Embedding Service - Local Chinese text embedding service"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cherry_bge_service.py .
COPY models ./models

# Expose port
EXPOSE 7860

# Set environment variables
ENV BGE_MODEL=/app/models/baai/bge-small-zh-v1.5
ENV BGE_CACHE_DIR=/app/models/baai/bge-small-zh-v1.5-cache
ENV BGE_PORT=7860
ENV BGE_HOST=0.0.0.0
ENV BGE_DEVICE=cpu
ENV BGE_BATCH_SIZE=32
ENV BGE_MAX_SEQ_LENGTH=512
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the service
CMD ["python", "cherry_bge_service.py"]
