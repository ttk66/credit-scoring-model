# Monitoring Stack - Quick Start Guide

## 🚀 Быстрый старт (5 минут)

### Без Slack интеграции:
```bash
bash scripts/deploy-monitoring.sh
```

### Со Slack интеграцией:
```bash
bash scripts/setup-monitoring-complete.sh "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

---

## 📊 Доступ к инструментам

### Prometheus (Metrics)
```bash
kubectl port-forward svc/prometheus 9090:9090 -n ml-serving &
# http://localhost:9090
```

**Полезные ссылки:**
- Targets: http://localhost:9090/targets
- Alerts: http://localhost:9090/alerts
- Graph: http://localhost:9090/graph

**Примеры запросов:**
```
rate(http_requests_total[1m])          # Requests per second
model_inference_duration_seconds       # Model latency
model_auc                              # Model accuracy
container_memory_usage_bytes{pod=~"api-.*"}  # Memory usage
```

### Grafana (Dashboards)
```bash
kubectl port-forward svc/grafana 3000:3000 -n ml-serving &
# http://localhost:3000
# Default: admin / admin (CHANGE IN PRODUCTION!)
```

**Доступные дашборды:**
- Model Performance (AUC, latency, throughput)
- Infrastructure Health (CPU, Memory, Network)
- Data Drift Monitoring (drift detection metrics)

### Loki (Logs)
```bash
kubectl port-forward svc/loki 3100:3100 -n ml-serving &
# Используйте в Grafana → Explore → Loki
```

**Примеры логирования:**
```
{namespace="ml-serving", container="api"}     # API logs
{pod_name="api-*", level="error"}             # Error logs
{container="postgres"}                         # DB logs
```

### Alertmanager (Alerts)
```bash
kubectl port-forward svc/alertmanager 9093:9093 -n ml-serving &
# http://localhost:9093
```

---

## 🔔 Алерты

Все алерты отправляются в Slack (если интегрирован):

| Alert | Condition | Channel |
|-------|-----------|---------|
| APIDown | API недоступен > 1 min | #critical-alerts 🚨 |
| HighErrorRate | Ошибки > 5% | #warnings ⚠️ |
| HighInferenceLatency | P95 > 1s | #warnings ⚠️ |
| DataDriftDetected | Дрифт обнаружен | #warnings ⚠️ |
| HighMemoryUsage | > 80% памяти | #warnings ⚠️ |
| ModelPerformanceDegradation | AUC < 0.75 | #warnings ⚠️ |

---

## 🔧 Конфигурация

### Prometheus
**Файл:** `kubernetes/monitoring/prometheus-config.yaml`

Измените:
- `scrape_interval` - частота сбора метрик
- `evaluation_interval` - частота проверки алертов
- `targets` - список источников метрик
- Alert rules - условия алертов

### Grafana
**Файл:** `kubernetes/monitoring/grafana-config.yaml`

Добавьте свои дашборды в секцию `data`:
```yaml
data:
  dashboard-provider.yaml: |
    ...
  your-dashboard.json: |
    {...}
```

### Loki
**Файл:** `kubernetes/monitoring/loki-logging.yaml`

Измените:
- `retention_period` - как долго хранить логи
- `ingestion_rate_mb` - максимальный размер логов в секунду
- Scrape configs - откуда собирать логи

### Alertmanager
**Файл:** `kubernetes/monitoring/alertmanager-config.yaml`

Требуется заполнить:
```bash
kubectl create secret generic alertmanager-secrets -n ml-serving \
  --from-literal=slack_webhook_url='https://hooks.slack.com/services/...'
```

---

## 📈 Метрики приложения

Добавьте в API (`src/api/app.py`):

```python
from prometheus_client import Counter, Histogram, Gauge

# Определить метрики
http_requests = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

inference_latency = Histogram(
    'model_inference_duration_seconds',
    'Model inference latency'
)

model_auc = Gauge('model_auc', 'Model AUC score')

# В обработчике запроса
@app.post("/predict")
async def predict(request: PredictRequest):
    # Record request
    http_requests.labels(
        method='POST',
        endpoint='/predict',
        status=200
    ).inc()
    
    # Record latency
    with inference_latency.time():
        prediction = model.predict(request.features)
    
    # Update AUC metric
    model_auc.set(current_model_auc)
    
    return {"prediction": prediction}

# Expose metrics endpoint
@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, REGISTRY
    return Response(generate_latest(REGISTRY), media_type="text/plain")
```

**Затем Prometheus автоматически соберёт метрики с `/metrics`**

---

## 🐛 Troubleshooting

### Prometheus не собирает метрики
```bash
# 1. Проверить targets
kubectl port-forward svc/prometheus 9090:9090 -n ml-serving
# http://localhost:9090/targets

# 2. Проверить что API доступен
kubectl port-forward svc/api-service 8000:8000 -n ml-serving
curl http://localhost:8000/metrics

# 3. Проверить logs Prometheus
kubectl logs deployment/prometheus -n ml-serving
```

### Grafana не видит Prometheus
```bash
# 1. Проверить что Prometheus работает
kubectl get pods -n ml-serving | grep prometheus

# 2. В Grafana: Configuration → Data Sources
# URL должен быть: http://prometheus:9090
# (используйте DNS имя сервиса, а не IP)

# 3. Проверить network connectivity
kubectl exec -it grafana-<POD> -n ml-serving -- \
  curl http://prometheus:9090
```

### Логи не собираются
```bash
# 1. Проверить Promtail pods
kubectl get ds -n ml-serving | grep promtail

# 2. Проверить что Promtail имеет доступ к логам
kubectl describe ds promtail -n ml-serving

# 3. Проверить Loki logs
kubectl logs ds/promtail -n ml-serving --tail=50

# 4. Проверить что Loki работает
kubectl exec -it loki-<POD> -n ml-serving -- \
  curl http://localhost:3100/ready
```

### Алерты не отправляются в Slack
```bash
# 1. Проверить что Alertmanager работает
kubectl get pods -n ml-serving | grep alertmanager

# 2. Проверить конфиг
kubectl get configmap alertmanager-config -n ml-serving -o yaml

# 3. Проверить что webhook URL правильный
kubectl get secret alertmanager-secrets -n ml-serving -o yaml

# 4. Проверить Alertmanager logs
kubectl logs deployment/alertmanager -n ml-serving

# 5. Trigger тестового алерта
kubectl port-forward svc/prometheus 9090:9090 -n ml-serving
# http://localhost:9090/graph → trigger manual alert
```

---

## 📋 Проверка здоровья

```bash
#!/bin/bash
# Check all monitoring components

NAMESPACE="ml-serving"

echo "Checking monitoring components..."
echo ""

# Check pods
echo "1. Monitoring Pods Status:"
kubectl get pods -n $NAMESPACE | grep -E "prometheus|grafana|loki|alertmanager|promtail"

# Check services
echo ""
echo "2. Monitoring Services:"
kubectl get svc -n $NAMESPACE | grep -E "prometheus|grafana|loki|alertmanager"

# Check Prometheus targets
echo ""
echo "3. Prometheus Targets (via API):"
kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE &>/dev/null &
sleep 2
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length' 2>/dev/null || echo "Prometheus not ready"

# Check Loki status
echo ""
echo "4. Loki Status:"
kubectl port-forward svc/loki 3100:3100 -n $NAMESPACE &>/dev/null &
sleep 2
curl -s http://localhost:3100/ready 2>/dev/null && echo "Loki: Ready" || echo "Loki: Not Ready"

# Check events
echo ""
echo "5. Recent Events:"
kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp' | tail -5

killall kubectl 2>/dev/null || true
```

Сохраните как `check-monitoring.sh` и запустите:
```bash
bash check-monitoring.sh
```

---

## 🔐 Security Best Practices

1. **Изменить пароль Grafana:**
   ```
   http://localhost:3000 → Profile → Change Password
   ```

2. **Ограничить доступ к Prometheus:**
   - Поставить reverse proxy с auth (nginx, Traefik)
   - Использовать network policies в K8s
   - Не expose напрямую в интернет

3. **Защитить Slack webhook:**
   - Хранить в K8s Secret (не в git)
   - Использовать restricted API token
   - Регулярно ротировать webhook

4. **Логирование доступа:**
   - Включить audit logs в Loki
   - Мониторить подозрительную активность
   - Настроить алерты на несанкционированный доступ

---

## 🚀 Масштабирование

### Для высоконагруженных систем:

**Prometheus:**
```yaml
# kubernetes/monitoring/prometheus-deployment.yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

**Grafana:**
```yaml
replicas: 3  # Multi-pod for HA
```

**Loki:**
```yaml
# Используйте distributed mode (не встроенный)
# https://grafana.com/docs/loki/latest/installation/
```

---

## 📚 Дополнительные ресурсы

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards)
- [Loki Best Practices](https://grafana.com/docs/loki/latest/best-practices/)
- [Kubernetes Monitoring](https://kubernetes.io/docs/tasks/debug-application-cluster/resource-metrics-pipeline/)

---

## ✅ Чек-лист мониторинга

- [ ] Все компоненты развёрнуты (`kubectl get pods -n ml-serving`)
- [ ] Prometheus собирает метрики (targets up)
- [ ] Grafana видит Prometheus как data source
- [ ] Grafana дашборды отображают данные
- [ ] Loki собирает логи
- [ ] Slack webhook настроен и работает
- [ ] Алерты тестировались (manual trigger)
- [ ] Runbook доступен ([kubernetes/monitoring/RUNBOOK.md](kubernetes/monitoring/RUNBOOK.md))

---

**Happy Monitoring! 📊**
