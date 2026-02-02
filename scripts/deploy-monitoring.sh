#!/bin/bash

# Deploy all monitoring and observability stack to Kubernetes
# Usage: bash scripts/deploy-monitoring.sh

set -e

NAMESPACE="ml-serving"
MONITORING_DIR="kubernetes/monitoring"

echo "=========================================="
echo "Deploying Monitoring & Observability Stack"
echo "=========================================="

# Check if namespace exists
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "Creating namespace $NAMESPACE..."
    kubectl create namespace $NAMESPACE
fi

# Deploy Prometheus
echo ""
echo "1. Deploying Prometheus..."
kubectl apply -f $MONITORING_DIR/prometheus-config.yaml
kubectl apply -f $MONITORING_DIR/prometheus-deployment.yaml
echo " Prometheus deployed"

# Wait for Prometheus to be ready
echo "Waiting for Prometheus to be ready..."
kubectl rollout status deployment/prometheus -n $NAMESPACE --timeout=120s

# Deploy Grafana
echo ""
echo "2. Deploying Grafana..."
kubectl apply -f $MONITORING_DIR/grafana-config.yaml
kubectl apply -f $MONITORING_DIR/grafana-deployment.yaml
echo " Grafana deployed"

# Wait for Grafana to be ready
echo "Waiting for Grafana to be ready..."
kubectl rollout status deployment/grafana -n $NAMESPACE --timeout=120s

# Deploy Loki + Promtail
echo ""
echo "3. Deploying Loki Logging Stack..."
kubectl apply -f $MONITORING_DIR/loki-logging.yaml
echo " Loki and Promtail deployed"

# Wait for Loki to be ready
echo "Waiting for Loki to be ready..."
kubectl rollout status deployment/loki -n $NAMESPACE --timeout=120s

# Verify all pods are running
echo ""
echo "4. Verifying all monitoring pods..."
kubectl get pods -n $NAMESPACE | grep -E "prometheus|grafana|loki|promtail"

# Port forwarding info
echo ""
echo "=========================================="
echo " Monitoring stack deployed successfully!"
echo "=========================================="
echo ""
echo "Access the monitoring tools:"
echo ""
echo "Prometheus:"
echo "  kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE"
echo "  URL: http://localhost:9090"
echo ""
echo "Grafana:"
echo "  kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
echo "  URL: http://localhost:3000"
echo "  Default credentials: admin/admin (CHANGE IN PRODUCTION!)"
echo ""
echo "Loki (in Grafana):"
echo "  Add data source: http://loki:3100"
echo ""
echo "Next steps:"
echo "  1. Port-forward to Grafana"
echo "  2. Add Prometheus as data source (http://prometheus:9090)"
echo "  3. Import dashboards from grafana-config.yaml"
echo "  4. Add Loki as data source (http://loki:3100)"
echo "  5. Set up alerting to Slack/Email"
echo ""
