#!/bin/bash

# Deployment Guide - Полный рабочий процесс развертывания
# Credit Scoring Model на Kubernetes в Yandex Cloud

set -e

# ============================================
# ФАЗА 1: ПОДГОТОВКА
# ============================================

echo "=== ФАЗА 1: ПОДГОТОВКА ==="

# 1. Установить зависимости
check_dependencies() {
    echo "Проверка зависимостей..."
    
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl не найден"; exit 1; }
    command -v yc >/dev/null 2>&1 || { echo "Yandex CLI не найден"; exit 1; }
    command -v docker >/dev/null 2>&1 || { echo "docker не найден"; exit 1; }
    command -v helm >/dev/null 2>&1 || { echo "helm не найден"; exit 1; }
    command -v kubeseal >/dev/null 2>&1 || { echo "kubeseal не найден (опционально)"; }
    
    echo "✅ Все зависимости установлены"
}

# 2. Настроить Yandex Cloud
setup_yandex_cloud() {
    echo "Настройка Yandex Cloud..."
    
    # Инициализировать Yandex CLI
    yc init --skip-tutorial
    
    # Установить default folder и cloud
    FOLDER_ID=$(yc config get folder-id)
    CLOUD_ID=$(yc config get cloud-id)
    
    echo "Cloud ID: $CLOUD_ID"
    echo "Folder ID: $FOLDER_ID"
}

# 3. Создать/обновить kubeconfig
setup_kubeconfig() {
    echo "Настройка kubeconfig..."
    
    CLUSTER_ID=$(yc managed-kubernetes cluster list --format json | jq -r '.[0].id')
    CLUSTER_NAME=$(yc managed-kubernetes cluster list --format json | jq -r '.[0].name')
    
    echo "Cluster: $CLUSTER_NAME ($CLUSTER_ID)"
    
    yc managed-kubernetes cluster get-credentials \
        --name $CLUSTER_NAME \
        --external \
        --force
    
    kubectl cluster-info
    kubectl get nodes
}

# ============================================
# ФАЗА 2: ПОДГОТОВКА DOCKER ОБРАЗОВ
# ============================================

echo ""
echo "=== ФАЗА 2: ПОДГОТОВКА DOCKER ОБРАЗОВ ==="

build_and_push_images() {
    echo "Сборка и загрузка Docker образов..."
    
    REGISTRY=$(yc container registry list --format json | jq -r '.[0].repository_name')
    echo "Registry: $REGISTRY"
    
    # API сервис
    echo "1. Сборка API образа..."
    docker build -t $REGISTRY/credit-scoring-api:1.0.0 \
        docker_for_nn_model/api/
    docker push $REGISTRY/credit-scoring-api:1.0.0
    
    # Frontend
    echo "2. Сборка Frontend образа..."
    docker build -t $REGISTRY/credit-scoring-frontend:1.0.0 \
        docker_for_nn_model/frontend/
    docker push $REGISTRY/credit-scoring-frontend:1.0.0
    
    # Data Loader
    echo "3. Сборка Data Loader образа..."
    docker build -t $REGISTRY/credit-scoring-data-loader:1.0.0 \
        docker_for_nn_model/data-loader/
    docker push $REGISTRY/credit-scoring-data-loader:1.0.0
    
    # Random Forest Backend
    echo "4. Сборка Random Forest Backend образа..."
    docker build -t $REGISTRY/credit-scoring-rf-backend:1.0.0 \
        docker_for_random_forest/
    docker push $REGISTRY/credit-scoring-rf-backend:1.0.0
    
    echo "✅ Все образы загружены в $REGISTRY"
}

# ============================================
# ФАЗА 3: НАСТРОЙКА KUBERNETES
# ============================================

echo ""
echo "=== ФАЗА 3: НАСТРОЙКА KUBERNETES ==="

setup_kubernetes() {
    echo "Создание namespace и RBAC..."
    
    # Создать namespace
    kubectl create namespace ml-serving || true
    kubectl label namespace ml-serving istio-injection=enabled || true
    
    # Создать Service Account
    kubectl create serviceaccount ml-serving-sa -n ml-serving || true
    
    # Создать ClusterRole и ClusterRoleBinding
    kubectl apply -f kubernetes/rbac/cluster-role.yaml 2>/dev/null || \
    cat << 'EOF' | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ml-serving-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get", "list"]
EOF
    
    kubectl create clusterrolebinding ml-serving-binding \
        --clusterrole=ml-serving-role \
        --serviceaccount=ml-serving:ml-serving-sa || true
}

# ============================================
# ФАЗА 4: СОЗДАНИЕ SECRETS И CONFIGMAPS
# ============================================

echo ""
echo "=== ФАЗА 4: SECRETS И CONFIGMAPS ==="

setup_secrets() {
    echo "Создание secrets и configmaps..."
    
    # Docker Registry Secret
    echo "1. Docker Registry Secret..."
    REGISTRY=$(yc container registry list --format json | jq -r '.[0].repository_name')
    OAUTH_TOKEN=$(yc iam create-token --service-account-name container-registry-user)
    
    kubectl create secret docker-registry docker-registry-credentials \
        --docker-server=cr.yandex \
        --docker-username=oauth \
        --docker-password=$OAUTH_TOKEN \
        --docker-email=system@example.com \
        -n ml-serving --dry-run=client -o yaml | kubectl apply -f -
    
    # Привязать secret к service account
    kubectl patch serviceaccount ml-serving-sa \
        -p '{"imagePullSecrets": [{"name": "docker-registry-credentials"}]}' \
        -n ml-serving
    
    # Database Secret
    echo "2. Database Secret..."
    kubectl create secret generic database-credentials \
        --from-literal=username=postgres \
        --from-literal=password=$(openssl rand -base64 32) \
        --from-literal=database=credit_scoring \
        -n ml-serving --dry-run=client -o yaml | kubectl apply -f -
    
    # Redis Secret
    echo "3. Redis Secret..."
    kubectl create secret generic redis-credentials \
        --from-literal=password=$(openssl rand -base64 32) \
        -n ml-serving --dry-run=client -o yaml | kubectl apply -f -
    
    # Storage/DVC Secret
    echo "4. Storage/DVC Secret..."
    kubectl create secret generic dvc-storage-credentials \
        --from-literal=access-key=your-access-key \
        --from-literal=secret-key=your-secret-key \
        --from-literal=endpoint=https://storage.yandexcloud.net \
        -n ml-serving --dry-run=client -o yaml | kubectl apply -f -
    
    echo "✅ Secrets созданы"
}

# ============================================
# ФАЗА 5: РАЗВЕРТЫВАНИЕ КОМПОНЕНТОВ
# ============================================

echo ""
echo "=== ФАЗА 5: РАЗВЕРТЫВАНИЕ КОМПОНЕНТОВ ==="

deploy_components() {
    echo "Развертывание компонентов..."
    
    # Применить ConfigMaps
    echo "1. ConfigMaps..."
    kubectl apply -f kubernetes/configs/all-configmaps.yaml
    
    # Применить Storage (PVC)
    echo "2. Persistent Storage..."
    kubectl apply -f kubernetes/storage/pvc-models.yaml
    
    # Применить Database
    echo "3. PostgreSQL..."
    kubectl apply -f kubernetes/deployments/postgresql-statefulset.yaml
    kubectl wait --for=condition=Ready pod/postgresql-0 -n ml-serving --timeout=300s
    
    # Применить Redis
    echo "4. Redis..."
    kubectl apply -f kubernetes/deployments/redis-statefulset.yaml
    kubectl wait --for=condition=Ready pod/redis-0 -n ml-serving --timeout=300s
    
    # Применить Services
    echo "5. Services..."
    kubectl apply -f kubernetes/services/all-services.yaml
    
    # Применить API
    echo "6. API..."
    kubectl apply -f kubernetes/deployments/api-deployment.yaml
    kubectl wait --for=condition=Ready pod -l app=credit-scoring-api -n ml-serving --timeout=300s
    
    # Применить Frontend
    echo "7. Frontend..."
    kubectl apply -f kubernetes/deployments/frontend-deployment.yaml
    kubectl wait --for=condition=Ready pod -l app=credit-scoring-frontend -n ml-serving --timeout=300s
    
    # Применить Celery Worker
    echo "8. Celery Worker..."
    kubectl apply -f kubernetes/deployments/celery-worker-deployment.yaml
    
    # Применить Data Loader Job
    echo "9. Data Loader Job..."
    kubectl apply -f kubernetes/jobs/data-loader-job.yaml
    
    # Применить Ingress
    echo "10. Ingress..."
    kubectl apply -f kubernetes/ingress/ingress.yaml
    kubectl apply -f kubernetes/ingress/frontend-ingress.yaml
    
    # Применить Network Policies
    echo "11. Network Policies..."
    kubectl apply -f kubernetes/network-policies/default-deny.yaml 2>/dev/null || true
    kubectl apply -f kubernetes/network-policies/*.yaml 2>/dev/null || true
    
    echo "✅ Компоненты развернуты"
}

# ============================================
# ФАЗА 6: ПРОВЕРКА РАЗВЕРТЫВАНИЯ
# ============================================

echo ""
echo "=== ФАЗА 6: ПРОВЕРКА РАЗВЕРТЫВАНИЯ ==="

verify_deployment() {
    echo "Проверка статуса развертывания..."
    
    echo ""
    echo "Pods:"
    kubectl get pods -n ml-serving -o wide
    
    echo ""
    echo "Services:"
    kubectl get svc -n ml-serving -o wide
    
    echo ""
    echo "Ingress:"
    kubectl get ingress -n ml-serving -o wide
    
    echo ""
    echo "PVC:"
    kubectl get pvc -n ml-serving -o wide
    
    echo ""
    echo "Events:"
    kubectl get events -n ml-serving --sort-by='.lastTimestamp' | tail -20
    
    # Проверить логи
    echo ""
    echo "Логи API:"
    kubectl logs -n ml-serving -l app=credit-scoring-api --tail=20 --timestamps=true 2>/dev/null || true
    
    echo ""
    echo "✅ Проверка завершена"
}

# ============================================
# ФАЗА 7: ТЕСТИРОВАНИЕ
# ============================================

echo ""
echo "=== ФАЗА 7: ТЕСТИРОВАНИЕ ==="

test_deployment() {
    echo "Тестирование развертывания..."
    
    # Port forward к API
    kubectl port-forward svc/credit-scoring-api 8000:8000 -n ml-serving &
    FORWARD_PID=$!
    sleep 2
    
    echo "1. Проверка Health endpoints..."
    
    # Test /health
    curl -s http://localhost:8000/health | jq . || true
    
    # Test /ready
    READY=$(curl -s http://localhost:8000/ready | jq -r '.status')
    if [ "$READY" == "ready" ]; then
        echo "✅ API ready"
    else
        echo "⚠️  API not ready"
    fi
    
    # Test /startup
    curl -s http://localhost:8000/startup | jq . || true
    
    # Test /predict
    echo ""
    echo "2. Тест предсказания..."
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{
            "features": [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.2, 0.1, 0.9],
            "model": "onnx"
        }' | jq . || true
    
    # Очистить port forward
    kill $FORWARD_PID 2>/dev/null || true
    
    echo ""
    echo "✅ Тестирование завершено"
}

# ============================================
# ФАЗА 8: POST-DEPLOYMENT
# ============================================

echo ""
echo "=== ФАЗА 8: POST-DEPLOYMENT ==="

post_deployment() {
    echo "Post-deployment настройка..."
    
    # Получить Ingress IP
    echo "1. Ingress адреса:"
    kubectl get ingress -n ml-serving -o wide
    
    # Получить LoadBalancer IP
    echo ""
    echo "2. LoadBalancer адреса:"
    kubectl get svc credit-scoring-api-lb -n ml-serving -o wide
    
    # Настроить DNS (если нужно)
    echo ""
    echo "3. DNS настройка (на ваше усмотрение):"
    INGRESS_IP=$(kubectl get ingress -n ml-serving credit-scoring-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "PENDING")
    echo "   Ingress IP: $INGRESS_IP"
    echo "   Add to /etc/hosts or DNS:"
    echo "   $INGRESS_IP api.credit-scoring.example.com"
    echo "   $INGRESS_IP app.credit-scoring.example.com"
    
    # Настроить мониторинг
    echo ""
    echo "4. Мониторинг (Prometheus + Grafana):"
    echo "   helm install prometheus prometheus-community/kube-prometheus-stack -n ml-serving"
    echo "   kubectl port-forward svc/prometheus-grafana 3000:80 -n ml-serving"
    
    # Настроить логирование
    echo ""
    echo "5. Логирование (ELK Stack):"
    echo "   helm install elasticsearch elastic/elasticsearch -n ml-serving"
    echo "   helm install kibana elastic/kibana -n ml-serving"
    
    # Backup sealing key (для SealedSecrets)
    echo ""
    echo "6. Backup Sealing Key (для Sealed Secrets):"
    mkdir -p backups
    kubectl get secret -n kube-system sealed-secrets-key -o yaml > backups/sealing-key-backup.yaml 2>/dev/null || true
    echo "   Сохранено: backups/sealing-key-backup.yaml"
    
    echo ""
    echo "✅ Post-deployment завершен"
}

# ============================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================

main() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   Credit Scoring Model - Deployment на Kubernetes             ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    check_dependencies
    setup_yandex_cloud
    setup_kubeconfig
    
    read -p "Продолжить с построением образов? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_and_push_images
    fi
    
    setup_kubernetes
    setup_secrets
    deploy_components
    verify_deployment
    
    read -p "Выполнить тесты? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_deployment
    fi
    
    post_deployment
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║   Развертывание завершено!                                    ║"
    echo "║   Проверьте: kubectl get pods -n ml-serving                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
}

# Запустить main функцию
main "$@"
