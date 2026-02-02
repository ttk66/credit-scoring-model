
FROM iterativeai/cml:0-dvc3-base AS dvc-base

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM mirror.gcr.io/library/python:3.11-slim AS loader

WORKDIR /app


RUN apt-get update && apt-get install -y \
    git \
    curl \
    ssh-client \
    && rm -rf /var/lib/apt/lists/*


COPY --from=dvc-base /opt/conda/bin/dvc /usr/local/bin/dvc
COPY --from=dvc-base /opt/conda/lib/python3.9/site-packages /usr/local/lib/python3.11/site-packages


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN useradd -m -u 1000 -s /bin/bash dataloader && \
    chown -R dataloader:dataloader /app
USER dataloader


COPY --chown=dataloader:dataloader . .


COPY dvc_config.yaml /home/dataloader/.config/dvc/config


RUN mkdir -p /app/data /app/.dvc/cache


RUN dvc init --no-scm


RUN dvc remote add -d s3remote s3://${DVC_BUCKET_NAME}/dvc-storage && \
    dvc config core.autostage true


ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DVC_BUCKET_NAME=credit-scoring-data \
    DVC_REMOTE=s3remote


CMD ["python", "scripts/load_data.py"]
