# 🏛️ Архитектура проекта - Credit Scoring Model

**Полное описание компонентов, их взаимодействия и технологий**

---

## 📐 Общая архитектура

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DEVELOPMENT LAYER                             │
│  Local Development → Git Commits → GitHub Repository (main branch)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         CI/CD AUTOMATION LAYER                          │
│                        GitHub Actions Pipeline                          │
│  ┌─────────────┬─────────────┬────────────────┬──────────────────────┐ │
│  │  1. Lint    │  2. Test    │  3. Security   │  4. Build & Push     │ │
│  │             │             │                │                      │ │
│  │ flake8      │ pytest       │ bandit         │ docker build         │ │
│  │ black       │ coverage     │ pip-audit      │ docker push          │ │
│  │ isort       │ integration  │ safety check   │ scan & sign images   │ │
│  └─────────────┴─────────────┴────────────────┴──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTAINER REGISTRY LAYER                             │
│             Yandex Container Registry (cr.yandex.io)                    │
│  ┌─────────────┬──────────────┬──────────────┬────────────────────┐   │
│  │ API Image   │ Frontend Img │ DataLoader   │ Random Forest Img  │   │
│  │ :v1.0.1     │ :v1.0.1      │ :v1.0.1      │ :v1.0.1            │   │
│  └─────────────┴──────────────┴──────────────┴────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│               INFRASTRUCTURE PROVISIONING LAYER (Terraform)             │
│                        Yandex Cloud Resources                           │
│  ┌──────────────┬─────────────┬──────────────┬──────────────────────┐ │
│  │ VPC Network  │ Subnets     │ Security Grp │ Load Balancer        │ │
│  │ (10.0.0.0/16)│ (3 zones)   │ (5 rules)    │ (External Traffic)   │ │
│  └──────────────┴─────────────┴──────────────┴──────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Kubernetes Cluster (K8s 1.26+)                                   │ │
│  │ ├─ Master nodes (managed by Yandex)                              │ │
│  │ ├─ Worker nodes (3 nodes, 4 CPU, 8 GB RAM each)                 │ │
│  │ └─ Persistent storage (SSD, 100 GB)                             │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Database (Managed PostgreSQL)                                    │ │
│  │ ├─ High availability (replicas)                                  │ │
│  │ ├─ Automated backups                                             │ │
│  │ └─ Encryption at rest                                            │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Object Storage (S3-compatible for models, data, artifacts)       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    KUBERNETES CLUSTER LAYER                             │
│                      (ml-serving namespace)                             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                    INGRESS & LOAD BALANCING                      │ │
│  │  ┌────────────────────────────────────────────────────────────┐ │ │
│  │  │ Ingress Controller (Nginx / Yandex Cloud ALB)             │ │ │
│  │  │ • api.credit-scoring.example.com → API Service            │ │ │
│  │  │ • app.credit-scoring.example.com → Frontend Service       │ │ │
│  │  │ • SSL/TLS (Let's Encrypt via cert-manager)                │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                  STATELESS SERVICES TIER                         │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐             │ │
│  │  │  API Deployment      │  │ Frontend Deployment  │             │ │
│  │  │  (3 replicas)        │  │ (2 replicas)         │             │ │
│  │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │             │ │
│  │  │ │  Pod 1           │ │  │ │  Pod 1           │ │             │ │
│  │  │ │ FastAPI + Models │ │  │ │ React + Nginx    │ │             │ │
│  │  │ │ Port: 8000       │ │  │ │ Port: 3000/80    │ │             │ │
│  │  │ └──────────────────┘ │  │ └──────────────────┘ │             │ │
│  │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │             │ │
│  │  │ │  Pod 2           │ │  │ │  Pod 2           │ │             │ │
│  │  │ │ (same as above)  │ │  │ │ (same as above)  │ │             │ │
│  │  │ └──────────────────┘ │  │ └──────────────────┘ │             │ │
│  │  │ ┌──────────────────┐ │  └──────────────────────┘             │ │
│  │  │ │  Pod 3           │ │                                        │ │
│  │  │ │ (same as above)  │ │                                        │ │
│  │  │ └──────────────────┘ │                                        │ │
│  │  │ HPA: 2-10 replicas   │                                        │ │
│  │  │ Trigger: CPU 80%     │                                        │ │
│  │  └──────────────────────┘                                        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │              STATEFUL SERVICES TIER (StatefulSet)               │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐             │ │
│  │  │  PostgreSQL          │  │  Redis Cache         │             │ │
│  │  │  (1 replica)         │  │  (1 replica)         │             │ │
│  │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │             │ │
│  │  │ │ postgres-0       │ │  │ │ redis-0          │ │             │ │
│  │  │ │ Port: 5432       │ │  │ │ Port: 6379       │ │             │ │
│  │  │ │ PVC: 10 GB       │ │  │ │ PVC: 5 GB        │ │             │ │
│  │  │ └──────────────────┘ │  │ └──────────────────┘ │             │ │
│  │  │ Backup: daily        │  │ Backup: disabled     │             │ │
│  │  └──────────────────────┘  └──────────────────────┘             │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │           BACKGROUND WORKERS (Celery + RabbitMQ)                │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │  Celery Worker Deployment (2 replicas)                   │   │ │
│  │  │  ├─ Prediction caching                                   │   │ │
│  │  │  ├─ Data preprocessing                                   │   │ │
│  │  │  ├─ Report generation                                    │   │ │
│  │  │  └─ Monitoring data collection                           │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │  Flower (Celery Monitoring Dashboard)                    │   │ │
│  │  │  http://flower.credit-scoring.local:5555                 │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │              JOBS & CRON JOBS (Data Processing)                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │  Data Loader Job (One-time)                              │   │ │
│  │  │  ├─ Loads data from Object Storage                       │   │ │
│  │  │  ├─ Validates with Great Expectations                    │   │ │
│  │  │  ├─ Stores in PostgreSQL                                 │   │ │
│  │  │  └─ Updates DVC cache                                    │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │  Data Loader CronJob (Daily at 2 AM)                     │   │ │
│  │  │  └─ Same as above, runs on schedule                      │   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │          CONFIGURATION MANAGEMENT (ConfigMaps & Secrets)         │ │
│  │                                                                   │ │
│  │  ConfigMaps:                                                     │ │
│  │  ├─ api-config          (API environment variables)             │ │
│  │  ├─ frontend-config     (Frontend settings)                     │ │
│  │  ├─ db-config           (Database parameters)                   │ │
│  │  └─ nginx-config        (Nginx configuration)                   │ │
│  │                                                                   │ │
│  │  Secrets (Kubernetes):                                           │ │
│  │  ├─ db-credentials      (postgres username/password)            │ │
│  │  ├─ storage-credentials (Object Storage access keys)            │ │
│  │  ├─ registry-credentials(Docker registry auth)                  │ │
│  │  └─ jwt-secret          (API JWT token signing)                 │ │
│  │                                                                   │ │
│  │  For Production: Sealed Secrets or HashiCorp Vault              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │              SECURITY (Network Policies & RBAC)                  │ │
│  │                                                                   │ │
│  │  Network Policies:                                               │ │
│  │  ├─ Ingress only from load balancer to API                      │ │
│  │  ├─ API to Database (port 5432 only)                            │ │
│  │  ├─ API to Redis (port 6379 only)                               │ │
│  │  ├─ Deny all by default, allow explicitly                       │ │
│  │  └─ Egress to external APIs only where needed                   │ │
│  │                                                                   │ │
│  │  RBAC (Role-Based Access Control):                               │ │
│  │  ├─ ServiceAccount: ml-serving-sa                               │ │
│  │  ├─ Role: ml-serving-role (read pods, logs)                     │ │
│  │  └─ RoleBinding: connects SA to Role                            │ │
│  │                                                                   │ │
│  │  Pod Security Policies:                                          │ │
│  │  ├─ Non-root users only                                         │ │
│  │  ├─ Read-only root filesystem                                   │ │
│  │  └─ No privileged containers                                    │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                RESOURCE MANAGEMENT & QUOTAS                      │ │
│  │                                                                   │ │
│  │  Requests (guaranteed):     Limits (max allowed):                │ │
│  │  ├─ API: 200m CPU / 256Mi   ├─ API: 1 CPU / 1 Gi               │ │
│  │  ├─ Frontend: 100m / 128Mi  ├─ Frontend: 500m / 512 Mi         │ │
│  │  ├─ Database: 500m / 1Gi    ├─ Database: 2 CPU / 4 Gi          │ │
│  │  └─ Redis: 100m / 128Mi     └─ Redis: 1 CPU / 1 Gi             │ │
│  │                                                                   │ │
│  │  Namespace Quotas:                                               │ │
│  │  ├─ Total CPU: 10 cores max                                     │ │
│  │  ├─ Total Memory: 16 GB max                                     │ │
│  │  └─ Total PVC: 100 GB max                                       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│               MONITORING & OBSERVABILITY LAYER                         │
│                      (ml-monitoring namespace)                         │
│  ┌────────────────┐  ┌─────────────┐  ┌────────────────┐              │
│  │ Prometheus     │  │   Grafana   │  │     Loki       │              │
│  │ (Metrics)      │  │(Dashboards) │  │   (Logs)       │              │
│  │ :9090          │  │ :3000       │  │ :3100          │              │
│  │                │  │             │  │                │              │
│  │ • Scrapes K8s  │  │ • 50+ panels│  │ • Promtail     │              │
│  │ • API metrics  │  │ • alerts    │  │ • queries      │              │
│  │ • Pod metrics  │  │ • trending  │  │ • retention    │              │
│  │ • Retention:   │  │             │  │ • retention:   │              │
│  │   15 days      │  │             │  │   7 days       │              │
│  └────────────────┘  └─────────────┘  └────────────────┘              │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Alertmanager                                                  │   │
│  │  ├─ Routes alerts (Slack, PagerDuty, Email)                    │   │
│  │  ├─ Deduplication                                              │   │
│  │  └─ Grouping                                                   │   │
│  └────────────────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Data Drift Detection (Evidently AI)                           │   │
│  │  ├─ Statistical tests                                          │   │
│  │  ├─ Data distribution shifts                                   │   │
│  │  └─ Triggers retraining                                        │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                  MODEL ORCHESTRATION LAYER (Airflow)                   │
│                                                                         │
│  DAG: credit_scoring_retrain (runs daily at 1 AM)                     │
│                                                                         │
│  ┌─────────────┐      ┌──────────────┐     ┌───────────────┐         │
│  │ 1. Fetch    │  →   │  2. Train    │  →  │ 3. Evaluate  │         │
│  │   Data      │      │   Models     │     │   Metrics    │         │
│  └─────────────┘      └──────────────┘     └───────────────┘         │
│                                                    ↓                   │
│  ┌─────────────┐      ┌──────────────┐     ┌───────────────┐         │
│  │ 6. Deploy   │  ←   │  5. Canary   │  ←  │ 4. A/B Test  │         │
│  │   Prod      │      │   Deploy     │     │   Compare    │         │
│  └─────────────┘      └──────────────┘     └───────────────┘         │
│                                                                         │
│  Each step:                                                            │
│  ├─ Logs to CloudWatch / ELK                                          │
│  ├─ Sends metrics to Prometheus                                       │
│  ├─ Updates MLFlow registry                                           │
│  └─ Alerts on failure                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Взаимодействие компонентов

### Запрос на предсказание (Flow)

```
User/Client
    ↓
    └─→ HTTPS Request (https://api.credit-scoring.example.com/predict)
           ↓
        Load Balancer (Yandex Cloud ALB)
           ↓
        Ingress Controller (Nginx)
           ↓
        K8s Service (API) ← Route traffic to available pods
           ↓
        Pod 1, Pod 2 или Pod 3 (whichever is ready)
           ↓
        FastAPI Application
           ├─→ Parse JSON request
           ├─→ Validate input (Pydantic)
           ├─→ Check Redis cache (optional)
           ├─→ Load model from memory (scikit-learn or ONNX)
           ├─→ Run inference
           ├─→ Cache result in Redis
           ├─→ Log to Loki
           └─→ Return JSON response with prediction + confidence
           ↓
        Response back to client
```

### Тренировка новой модели (Flow)

```
Scheduled (Airflow at 1 AM)
    ↓
DAG: credit_scoring_retrain starts
    ↓
Task 1: Fetch Data
├─ Connect to PostgreSQL
├─ Load new data (since last run)
├─ Check with Great Expectations
└─ Store in /data/processed/

    ↓
Task 2: Train Models
├─ Random Forest
│  └─ 30 min training → best_model_new.joblib
├─ Neural Network
│  └─ 15 min training → nn_model_new.pth
└─ Convert to ONNX
    └─ nn_model_new.onnx (for optimization)

    ↓
Task 3: Evaluate & Compare
├─ Load previous models
├─ Compare metrics (AUC, F1, precision, recall)
├─ Run A/B test on validation set
└─ Evidently AI: Check for data drift

    ↓
Task 4: Model Registry (MLFlow)
├─ Register new model versions
├─ Tag with timestamp + metrics
└─ Store in Object Storage

    ↓
Task 5: Canary Deploy (10% traffic)
├─ Create new deployment with 10% replicas
├─ Monitor for 24 hours
├─ Check: latency, errors, predictions
└─ Alert if degradation detected

    ↓
Task 6: Blue-Green Deploy (100% traffic)
├─ Gradually shift traffic (10% → 25% → 50% → 100%)
├─ Roll back if issues detected
├─ Update DNS + service selector
└─ Send Slack notification: "Model updated"
```

### Обнаружение дрифта данных (Flow)

```
Continuously (every 1 hour)
    ↓
Evidently AI service
    ├─ Query recent predictions from PostgreSQL
    ├─ Fetch original training data distribution
    │
    ├─ Statistical tests:
    │  ├─ Kolmogorov-Smirnov test (numerical)
    │  ├─ Chi-square test (categorical)
    │  └─ Population Stability Index
    │
    ├─ If drift detected:
    │  ├─ Calculate drift magnitude
    │  ├─ Send alert to Slack
    │  ├─ Log to Prometheus (metric: data_drift_detected = 1)
    │  ├─ Trigger Airflow DAG if threshold exceeded
    │  └─ Create incident in monitoring system
    │
    └─ Store metrics in Prometheus + Grafana dashboard
```

---

## 🛠️ Компоненты по слоям

### 1️⃣ DEVELOPMENT LAYER

| Компонент | Технология | Назначение |
|-----------|-----------|-----------|
| IDE | VSCode / PyCharm | Разработка |
| Version Control | Git + GitHub | Control версий |
| Local Runtime | Python 3.9+ | Выполнение кода |
| Package Mgmt | pip + venv | Зависимости |

### 2️⃣ TESTING & QUALITY LAYER

| Компонент | Технология | Назначение |
|-----------|-----------|-----------|
| Unit Tests | pytest | Модульное тестирование |
| Code Coverage | pytest-cov | Покрытие кода |
| Linting | flake8, black | Стиль кода |
| Static Analysis | bandit, Pylint | Качество кода |
| Dependency Check | pip-audit, safety | Безопасность |
| Integration Test | pytest | Интеграционные тесты |

### 3️⃣ CI/CD LAYER

| Компонент | Технология | Назначение |
|-----------|-----------|-----------|
| Pipeline | GitHub Actions | Автоматизация |
| Build | Docker | Контейнеризация |
| Registry | Yandex Container Registry | Хранилище образов |
| Scanning | Trivy, Cosign | Сканирование образов |

### 4️⃣ INFRASTRUCTURE LAYER

| Компонент | Технология | Назначение |
|-----------|-----------|-----------|
| IaC | Terraform | Инфраструктура |
| Cloud Provider | Yandex Cloud | Облако |
| Networking | VPC, Subnets, Security Groups | Сеть |
| Database | PostgreSQL (Managed) | Персистентное хранилище |
| Object Storage | Yandex S3 (Object Storage) | Модели, артефакты |

### 5️⃣ CONTAINER ORCHESTRATION LAYER

| Компонент | Технология | Назначение |
|-----------|-----------|-----------|
| Orchestrator | Kubernetes 1.26 | Оркестрация контейнеров |
| API Server | FastAPI | REST API |
| Frontend | React + Nginx | Веб интерфейс |
| Task Queue | Celery + RabbitMQ | Фоновые задачи |
| Caching | Redis | In-memory cache |
| Ingress | Nginx Ingress | Входящий трафик |
| Service Mesh | (optional) Istio | Микросервисная коммуникация |

### 6️⃣ MONITORING & OBSERVABILITY

| Компонент | Технология | Назначение |
|-----------|-----------|-----------|
| Metrics | Prometheus | Сбор метрик |
| Visualization | Grafana | Визуализация |
| Logs | Loki + Promtail | Централизованное логирование |
| Alerts | Alertmanager | Управление алертами |
| Distributed Tracing | Jaeger (optional) | Трейсинг запросов |
| Health Checks | Custom endpoints | Проверка здоровья |

### 7️⃣ ML PIPELINE LAYER

| Компонент | Технология | Назначение |
|-----------|-----------|-----------|
| Data Versioning | DVC | Управление данными |
| Data Validation | Great Expectations | Валидация данных |
| Feature Store | pandas + pickle | Фичи модели |
| Model Training | scikit-learn, PyTorch | Обучение |
| Model Optimization | ONNX, quantization | Оптимизация |
| Model Registry | MLFlow | Версионирование моделей |
| Drift Detection | Evidently AI | Обнаружение дрифта |
| A/B Testing | Custom logic | Сравнение моделей |
| Orchestration | Airflow | Автоматизация pipeline |

### 8️⃣ SECURITY LAYER

| Компонент | Технология | Назначение |
|-----------|-----------|-----------|
| Network Policy | K8s Network Policies | Сетевая безопасность |
| RBAC | K8s RBAC | Доступ контроль |
| Secrets Mgmt | Sealed Secrets / Vault | Управление секретами |
| SSL/TLS | cert-manager + Let's Encrypt | Шифрование |
| Image Scanning | Trivy, Cosign | Сканирование образов |
| Vulnerability Scanning | bandit, pip-audit | Уязвимости |

---

## 🔌 API Endpoints

### Основные endpoints

```
POST /predict
├─ Input: JSON with features
├─ Output: prediction + confidence + model_version
└─ Auth: Optional API key

GET /health
├─ Input: none
├─ Output: { "status": "ok" }
└─ Used by: load balancer

GET /ready
├─ Input: none
├─ Output: { "database": "ok", "redis": "ok", "model": "ok" }
└─ Used by: K8s readiness probe

GET /live
├─ Input: none
├─ Output: { "status": "alive" }
└─ Used by: K8s liveness probe

GET /metrics
├─ Input: none
├─ Output: Prometheus metrics
└─ Scraped by: Prometheus every 30 seconds

GET /docs
├─ Input: none
├─ Output: Swagger UI
└─ URL: https://api.credit-scoring.example.com/docs

GET /redoc
├─ Input: none
├─ Output: ReDoc (OpenAPI documentation)
└─ URL: https://api.credit-scoring.example.com/redoc
```

---

## 📊 Количественные показатели

### Масштаб и производительность

| Метрика | Значение |
|---------|----------|
| Requests/second | 100-500 (depends on model size) |
| Latency p95 | < 200ms |
| Latency p99 | < 500ms |
| Uptime SLA | 99.5% (4.38 hours downtime/month) |
| Model size | RF: 50MB, NN: 100MB, ONNX: 80MB |
| Inference time | RF: 1-5ms, NN: 5-20ms |
| API CPU per pod | 200m (0.2 core) |
| API Memory per pod | 512MB |
| Database CPU | 1 core (managed service) |
| Database Storage | 100GB |
| Object Storage | 500GB (models + backups) |

### Затраты

| Компонент | Стоимость/месяц |
|-----------|----------------|
| K8s кластер (3 nodes) | ~$50-100 |
| Database (PostgreSQL) | ~$20-50 |
| Object Storage | ~$5-20 |
| Load Balancer | ~$5-10 |
| VPC & Networking | ~$5 |
| Monitoring | ~$10-20 |
| **ИТОГО** | **~$100-200/month** |

---

## 🔐 Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────┐
│  Layer 1: Web Application Firewall       │
│  (DDoS protection, rate limiting)       │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Layer 2: TLS/SSL Encryption             │
│  (HTTPS only, no HTTP allowed)          │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Layer 3: Ingress & Load Balancer Auth  │
│  (API key validation, JWT token)        │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Layer 4: Network Policies               │
│  (K8s network segmentation)              │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Layer 5: RBAC (Role-Based Access)       │
│  (ServiceAccounts, Roles, RoleBindings)  │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Layer 6: Pod Security                   │
│  (Non-root, read-only FS, no privileges) │
└─────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│  Layer 7: Secrets Management             │
│  (Sealed Secrets, encryption at rest)    │
└─────────────────────────────────────────┘
```

---

## 📈 Масштабирование

### Горизонтальное масштабирование (HPA)

```yaml
# API auto-scales from 2 to 10 replicas based on metrics:
- CPU: > 80% → scale up
- Memory: > 80% → scale up
- Custom metrics: predictions/second

Example:
Initial: 2 replicas
Load: 100 req/sec → 5 replicas
Load: 200 req/sec → 10 replicas
Load: 50 req/sec → 2 replicas (scales down after 5 min)
```

### Вертикальное масштабирование

```yaml
# Node resources (per Yandex Cloud instance):
- CPU: 4 cores
- Memory: 8 GB
- Storage: 100 GB SSD

# To scale vertically:
1. Create new nodes with more resources
2. Drain old nodes (reschedule pods)
3. Delete old nodes
4. Terraform automatically handles this
```

---

## 🚀 Deployment Strategies

### Canary Deployment (10% traffic)

```
Initial:
  blue: 90% traffic (current stable version)
  green: 10% traffic (new version)

  Monitor for 24 hours...
  
  If OK:
    blue: 50%
    green: 50%
    
    Monitor for 12 hours...
    
    If still OK:
      blue: 0% (delete)
      green: 100% (stable)
      
  If issues:
    blue: 100% (restore)
    green: 0% (rollback)
```

### Blue-Green Deployment

```
Blue (current):
  5 replicas
  Service selector: version=blue
  
Green (new):
  5 replicas
  Service selector: version=green
  
Switch traffic:
  Service selector: version=green (instant switch)
  
If issues:
  Service selector: version=blue (instant rollback)
```

---

## 🎛️ Monitoring & Alerts

### Key Metrics

```
Application:
├─ api_requests_total (requests per second)
├─ api_request_duration_seconds (histogram: p50, p95, p99)
├─ api_errors_total (4xx, 5xx errors)
├─ model_inference_time_seconds (prediction latency)
└─ cache_hit_ratio (Redis effectiveness)

Infrastructure:
├─ node_cpu_usage (%)
├─ node_memory_usage (%)
├─ pod_cpu_usage (cores)
├─ pod_memory_usage (MB)
├─ pvc_usage (% of storage)
└─ network_io (bytes/sec)

Database:
├─ pg_queries_per_second
├─ pg_connection_count
├─ pg_cache_hit_ratio
└─ pg_replication_lag (if HA)

Data Quality:
├─ data_drift_detected (0 or 1)
├─ feature_missing_ratio (%)
└─ data_validation_failures (count)
```

### Alert Rules

```
Warning:
- API latency p95 > 200ms
- Error rate > 1%
- Pod memory > 80% of limit
- CPU usage > 70%

Critical:
- API latency p95 > 1s
- Error rate > 5%
- Pod memory > 95% of limit
- Database connection pool exhausted
- Pod CrashLoopBackOff
- Data drift magnitude > 0.5
```

---

## 📚 Технологический стек (Summary)

```
Frontend:
├─ React.js / Vue.js
├─ Nginx (reverse proxy)
└─ HTTPS (Let's Encrypt)

Backend:
├─ Python 3.9+
├─ FastAPI (async framework)
├─ Pydantic (validation)
├─ SQLAlchemy (ORM)
├─ Celery (task queue)
└─ Redis (caching)

ML/Data:
├─ scikit-learn (Random Forest)
├─ PyTorch (Neural Networks)
├─ ONNX (model format)
├─ pandas (data processing)
├─ Great Expectations (data validation)
├─ Evidently AI (drift detection)
├─ DVC (data versioning)
└─ MLFlow (model registry)

DevOps:
├─ Docker (containerization)
├─ Kubernetes (orchestration)
├─ Terraform (IaC)
├─ Yandex Cloud (cloud provider)
├─ Airflow (ML orchestration)
└─ GitHub Actions (CI/CD)

Monitoring:
├─ Prometheus (metrics)
├─ Grafana (visualization)
├─ Loki (logging)
├─ Alertmanager (alerts)
└─ Jaeger (tracing, optional)

Security:
├─ SSL/TLS (Let's Encrypt)
├─ Sealed Secrets / Vault
├─ Network Policies
├─ RBAC
├─ Pod Security Policies
└─ Trivy (image scanning)
```

---

**Это полная архитектура production-ready системы для ML-приложения!** 🎉

