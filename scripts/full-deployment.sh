#!/bin/bash

# Deployment Guide -    
# Credit Scoring Model  Kubernetes  Yandex Cloud

set -e

# ============================================
#  1: 
# ============================================

echo "===  1:  ==="

# 1.  
check_dependencies() {
    echo " ..."
    
    command -v kubectl >/dev/null 2>&1 || { echo "kubectl  "; exit 1; }
    command -v yc >/dev/null 2>&1 || { echo "Yandex CLI  "; exit 1; }
    command -v docker >/dev/null 2>&1 || { echo "docker  "; exit 1; }
    command -v helm >/dev/null 2>&1 || { echo "helm  "; exit 1; }
    command -v kubeseal >/dev/null 2>&1 || { echo "kubeseal   ()"; }
    
    echo "   "
}

# 2.  Yandex Cloud
setup_yandex_cloud() {
    echo " Yandex Cloud..."
    
    #  Yandex CLI
    yc init --skip-tutorial
    
    #  default folder  cloud
    FOLDER_ID=$(yc config get folder-id)
    CLOUD_ID=$(yc config get cloud-id)
    
    echo "Cloud ID: $CLOUD_ID"
    echo "Folder ID: $FOLDER_ID"
}

# 3. / kubeconfig
setup_kubeconfig() {
    echo " kubeconfig..."
    
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
#  2:  DOCKER 
# ============================================

echo ""
echo "===  2:  DOCKER  ==="

build_and_push_images() {
    echo "   Docker ..."
    
    REGISTRY=$(yc container registry list --format json | jq -r '.[0].repository_name')
    echo "Registry: $REGISTRY"
    
    # API 
    echo "1.  API ..."
    docker build -t $REGISTRY/credit-scoring-api:1.0.0 \
        docker_for_nn_model/api/
    docker push $REGISTRY/credit-scoring-api:1.0.0
    
    # Frontend
    echo "2.  Frontend ..."
    docker build -t $REGISTRY/credit-scoring-frontend:1.0.0 \
        docker_for_nn_model/frontend/
    docker push $REGISTRY/credit-scoring-frontend:1.0.0
    
    # Data Loader
    echo "3.  Data Loader ..."
    docker build -t $REGISTRY/credit-scoring-data-loader:1.0.0 \
        docker_for_nn_model/data-loader/
    docker push $REGISTRY/credit-scoring-data-loader:1.0.0
    
    # Random Forest Backend
    echo "4.  Random Forest Backend ..."
    docker build -t $REGISTRY/credit-scoring-rf-backend:1.0.0 \
        docker_for_random_forest/
    docker push $REGISTRY/credit-scoring-rf-backend:1.0.0
    
    echo "     $REGISTRY"
}

# ============================================
#  3:  KUBERNETES
# ============================================

echo ""
echo "===  3:  KUBERNETES ==="

setup_kubernetes() {
    echo " namespace  RBAC..."
    
    #  namespace
    kubectl create namespace ml-serving || true
    kubectl label namespace ml-serving istio-injection=enabled || true
    
    #  Service Account
    kubectl create serviceaccount ml-serving-sa -n ml-serving || true
    
    #  ClusterRole  ClusterRoleBinding
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
#  4:  SECRETS  CONFIGMAPS
# ============================================

echo ""
echo "===  4: SECRETS  CONFIGMAPS ==="

setup_secrets() {
    echo " secrets  configmaps..."
    
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
    
    #  secret  service account
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
    
    echo " Secrets "
}

# ============================================
#  5:  
# ============================================

echo ""
echo "===  5:   ==="

deploy_components() {
    echo " ..."
    
    #  ConfigMaps
    echo "1. ConfigMaps..."
    kubectl apply -f kubernetes/configs/all-configmaps.yaml
    
    #  Storage (PVC)
    echo "2. Persistent Storage..."
    kubectl apply -f kubernetes/storage/pvc-models.yaml
    
    #  Database
    echo "3. PostgreSQL..."
    kubectl apply -f kubernetes/deployments/postgresql-statefulset.yaml
    kubectl wait --for=condition=Ready pod/postgresql-0 -n ml-serving --timeout=300s
    
    #  Redis
    echo "4. Redis..."
    kubectl apply -f kubernetes/deployments/redis-statefulset.yaml
    kubectl wait --for=condition=Ready pod/redis-0 -n ml-serving --timeout=300s
    
    #  Services
    echo "5. Services..."
    kubectl apply -f kubernetes/services/all-services.yaml
    
    #  API
    echo "6. API..."
    kubectl apply -f kubernetes/deployments/api-deployment.yaml
    kubectl wait --for=condition=Ready pod -l app=credit-scoring-api -n ml-serving --timeout=300s
    
    #  Frontend
    echo "7. Frontend..."
    kubectl apply -f kubernetes/deployments/frontend-deployment.yaml
    kubectl wait --for=condition=Ready pod -l app=credit-scoring-frontend -n ml-serving --timeout=300s
    
    #  Celery Worker
    echo "8. Celery Worker..."
    kubectl apply -f kubernetes/deployments/celery-worker-deployment.yaml
    
    #  Data Loader Job
    echo "9. Data Loader Job..."
    kubectl apply -f kubernetes/jobs/data-loader-job.yaml
    
    #  Ingress
    echo "10. Ingress..."
    kubectl apply -f kubernetes/ingress/ingress.yaml
    kubectl apply -f kubernetes/ingress/frontend-ingress.yaml
    
    #  Network Policies
    echo "11. Network Policies..."
    kubectl apply -f kubernetes/network-policies/default-deny.yaml 2>/dev/null || true
    kubectl apply -f kubernetes/network-policies/*.yaml 2>/dev/null || true
    
    echo "  "
}

# ============================================
#  6:  
# ============================================

echo ""
echo "===  6:   ==="

verify_deployment() {
    echo "  ..."
    
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
    
    #  
    echo ""
    echo " API:"
    kubectl logs -n ml-serving -l app=credit-scoring-api --tail=20 --timestamps=true 2>/dev/null || true
    
    echo ""
    echo "  "
}

# ============================================
#  7: 
# ============================================

echo ""
echo "===  7:  ==="

test_deployment() {
    echo " ..."
    
    # Port forward  API
    kubectl port-forward svc/credit-scoring-api 8000:8000 -n ml-serving &
    FORWARD_PID=$!
    sleep 2
    
    echo "1.  Health endpoints..."
    
    # Test /health
    curl -s http://localhost:8000/health | jq . || true
    
    # Test /ready
    READY=$(curl -s http://localhost:8000/ready | jq -r '.status')
    if [ "$READY" == "ready" ]; then
        echo " API ready"
    else
        echo "  API not ready"
    fi
    
    # Test /startup
    curl -s http://localhost:8000/startup | jq . || true
    
    # Test /predict
    echo ""
    echo "2.  ..."
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{
            "features": [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.2, 0.1, 0.9],
            "model": "onnx"
        }' | jq . || true
    
    #  port forward
    kill $FORWARD_PID 2>/dev/null || true
    
    echo ""
    echo "  "
}

# ============================================
#  8: POST-DEPLOYMENT
# ============================================

echo ""
echo "===  8: POST-DEPLOYMENT ==="

post_deployment() {
    echo "Post-deployment ..."
    
    #  Ingress IP
    echo "1. Ingress :"
    kubectl get ingress -n ml-serving -o wide
    
    #  LoadBalancer IP
    echo ""
    echo "2. LoadBalancer :"
    kubectl get svc credit-scoring-api-lb -n ml-serving -o wide
    
    #  DNS ( )
    echo ""
    echo "3. DNS  (  ):"
    INGRESS_IP=$(kubectl get ingress -n ml-serving credit-scoring-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "PENDING")
    echo "   Ingress IP: $INGRESS_IP"
    echo "   Add to /etc/hosts or DNS:"
    echo "   $INGRESS_IP api.credit-scoring.example.com"
    echo "   $INGRESS_IP app.credit-scoring.example.com"
    
    #  
    echo ""
    echo "4.  (Prometheus + Grafana):"
    echo "   helm install prometheus prometheus-community/kube-prometheus-stack -n ml-serving"
    echo "   kubectl port-forward svc/prometheus-grafana 3000:80 -n ml-serving"
    
    #  
    echo ""
    echo "5.  (ELK Stack):"
    echo "   helm install elasticsearch elastic/elasticsearch -n ml-serving"
    echo "   helm install kibana elastic/kibana -n ml-serving"
    
    # Backup sealing key ( SealedSecrets)
    echo ""
    echo "6. Backup Sealing Key ( Sealed Secrets):"
    mkdir -p backups
    kubectl get secret -n kube-system sealed-secrets-key -o yaml > backups/sealing-key-backup.yaml 2>/dev/null || true
    echo "   : backups/sealing-key-backup.yaml"
    
    echo ""
    echo " Post-deployment "
}

# ============================================
#  
# ============================================

main() {
    echo ""
    echo "   Credit Scoring Model - Deployment  Kubernetes             "
    echo ""
    echo ""
    
    check_dependencies
    setup_yandex_cloud
    setup_kubeconfig
    
    read -p "   ? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_and_push_images
    fi
    
    setup_kubernetes
    setup_secrets
    deploy_components
    verify_deployment
    
    read -p " ? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_deployment
    fi
    
    post_deployment
    
    echo ""
    echo ""
    echo "    !                                    "
    echo "   : kubectl get pods -n ml-serving                   "
    echo ""
}

#  main 
main "$@"
