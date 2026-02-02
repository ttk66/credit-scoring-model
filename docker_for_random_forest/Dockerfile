FROM python:3.10-slim AS builder

WORKDIR /app

# Установка build-зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements.txt и установка пакетов
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Production stage
FROM python:3.10-slim AS runtime

WORKDIR /app

# Установка runtime-зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование установленных пакетов из builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Создание non-root пользователя
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Копирование приложения
COPY --chown=appuser:appuser . .

# Создание необходимых директорий
RUN mkdir -p /app/data/processed /app/models /app/mlruns /app/logs

# Установка переменных окружения
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/best_model.joblib \
    PORT=8000

# Экспонирование порта
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск приложения
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]