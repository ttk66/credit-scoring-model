FROM mirror.gcr.io/library/python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir boto3

RUN useradd -m -u 1000 -s /bin/bash downloader && \
    chown -R downloader:downloader /app
USER downloader

COPY --chown=downloader:downloader download_model.py /app/download_model.py

ENV PYTHONUNBUFFERED=1 \
    MODEL_KEY=credit_scoring.onnx \
    MODEL_PATH=/models/credit_scoring.onnx \
    STORAGE_ENDPOINT=https://storage.yandexcloud.net

CMD ["python", "/app/download_model.py"]
