FROM mirror.gcr.io/library/python:3.11 AS builder

WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


COPY docker_for_nn_model/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt



FROM mirror.gcr.io/library/python:3.11-slim AS runtime

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY --from=builder /usr/local /usr/local


RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser


COPY --chown=appuser:appuser src /app/src


RUN mkdir -p /app/data /app/models /app/logs


ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000


EXPOSE 8000


HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


ENTRYPOINT ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
