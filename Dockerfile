FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/processed /app/models /app/mlruns

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/best_model.joblib

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]