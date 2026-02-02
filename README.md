# Credit Scoring Model

## 1. О проекте

Проект реализует полный MLOps цикл для скоринга кредитного риска:

- обучение и оптимизация моделей;
- конвертация модели в ONNX и инференс через FastAPI;
- инфраструктура в Yandex Cloud через Terraform и Kubernetes;
- CI/CD в GitHub Actions;
- мониторинг (Prometheus, Grafana, Loki, Alertmanager);
- детектирование дрифта и автоматизация переобучения через Airflow.

## 2. Структура репозитория

- `src/` - код API, моделей, фичей и мониторинга;
- `tests/` - тесты;
- `models/` - артефакты моделей;
- `docker_for_nn_model/` - Dockerfile для API, frontend, model-downloader;
- `infra/` - Terraform для облачной инфраструктуры;
- `kubernetes/` - Kubernetes манифесты;
- `airflow/dags/` - DAG для переобучения;
- `.github/workflows/` - CI/CD пайплайны.

## 3. Требования

- Python 3.11+;
- Docker;
- kubectl;
- Terraform 1.5+;
- Yandex Cloud CLI (`yc`);
- доступ в Yandex Container Registry и Managed Kubernetes.

## 4. Локальный запуск (минимум)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests
```

Запуск API локально:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

## 5. Сборка и публикация образов

Пример для вашего реестра:

```bash
docker build -t cr.yandex/<registry_id>/credit-scoring-api:latest -f docker_for_nn_model/api/Dockerfile .
docker push cr.yandex/<registry_id>/credit-scoring-api:latest

docker build -t cr.yandex/<registry_id>/model-downloader:latest -f docker_for_nn_model/model-downloader/Dockerfile .
docker push cr.yandex/<registry_id>/model-downloader:latest
```

## 6. Развертывание инфраструктуры (Terraform)

```bash
cd infra
terraform init
terraform plan
terraform apply
```

Важно:

- рабочий `terraform.tfvars` должен быть в `infra/`;
- не храните токены и ключи в Git;
- для Yandex провайдера должен быть валидный `yc_token` или service account key.

## 7. Развертывание в Kubernetes

Применение базовых манифестов:

```bash
kubectl apply -f kubernetes/namespaces/ml-serving.yaml
kubectl apply -f kubernetes/configs/all-configmaps.yaml
kubectl apply -f kubernetes/services/all-services.yaml
kubectl apply -f kubernetes/deployments/postgresql-statefulset.yaml
kubectl apply -f kubernetes/deployments/redis-statefulset.yaml
kubectl apply -f kubernetes/deployments/api-deployment.yaml
kubectl apply -f kubernetes/ingress/ingress.yaml
```

Проверка:

```bash
kubectl get pods -n ml-serving
kubectl get svc -n ml-serving
kubectl get ingress -n ml-serving
```

## 8. Мониторинг и observability

Применение мониторинга:

```bash
kubectl apply -f kubernetes/monitoring/prometheus-config.yaml
kubectl apply -f kubernetes/monitoring/prometheus-deployment.yaml
kubectl apply -f kubernetes/monitoring/grafana-config.yaml
kubectl apply -f kubernetes/monitoring/grafana-deployment.yaml
kubectl apply -f kubernetes/monitoring/loki-logging.yaml
kubectl apply -f kubernetes/monitoring/alertmanager-config.yaml
```

Порт-форвард:

```bash
kubectl port-forward svc/prometheus 9090:9090 -n ml-serving
kubectl port-forward svc/grafana 3000:3000 -n ml-serving
kubectl port-forward svc/loki 3100:3100 -n ml-serving
```

## 9. Дрифт и переобучение

API экспортирует метрики:

- `data_drift_detected`;
- `evidently_feature_drift_score`;
- `evidently_prediction_drift_score`;
- `evidently_target_drift_score`;
- `model_performance_decay`;
- `concept_drift_detected`.

Airflow DAG:

- файл: `airflow/dags/credit_scoring_retrain.py`;
- регулярный запуск по cron;
- триггеры: изменение данных и метрика дрифта.

Рекомендуемые Airflow Variables:

- `retrain_data_path` (например `/opt/airflow/data/processed/credit_data_features.csv`);
- `prometheus_url` (например `http://prometheus:9090`);
- `last_retrain_data_mtime` (обновляется автоматически после успешного цикла).

## 10. CI/CD

Основной пайплайн: `.github/workflows/ci-cd-pipeline.yaml`

Этапы:

- build;
- test;
- security;
- deploy (staging/production);
- monitor.

Нужные секреты GitHub:

- `YC_REGISTRY_ID`;
- `YC_OAUTH_TOKEN`;
- `KUBE_CONFIG_STAGING`;
- `KUBE_CONFIG_PRODUCTION`.


