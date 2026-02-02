#!/bin/bash

# Complete monitoring setup with all components
# Usage: bash scripts/setup-monitoring-complete.sh [slack-webhook-url] [pagerduty-key]

set -e

NAMESPACE="ml-serving"
MONITORING_DIR="kubernetes/monitoring"
SLACK_WEBHOOK=${1:-""}
PAGERDUTY_KEY=${2:-""}

echo "=========================================="
echo "Complete Monitoring Stack Setup"
echo "=========================================="

# 1. Create monitoring directory if doesn't exist
echo ""
echo "1. Creating monitoring namespace..."
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    kubectl create namespace $NAMESPACE
    echo "✓ Namespace created"
else
    echo "✓ Namespace already exists"
fi

# 2. Setup Alertmanager secrets (if webhook provided)
if [ -n "$SLACK_WEBHOOK" ]; then
    echo ""
    echo "2. Creating Alertmanager secrets..."
    kubectl create secret generic alertmanager-secrets \
      -n $NAMESPACE \
      --from-literal=slack_webhook_url="$SLACK_WEBHOOK" \
      --from-literal=pagerduty_service_key="${PAGERDUTY_KEY:-}" \
      --dry-run=client -o yaml | kubectl apply -f -
    echo "✓ Alertmanager secrets created"
fi

# 3. Deploy Prometheus
echo ""
echo "3. Deploying Prometheus..."
kubectl apply -f $MONITORING_DIR/prometheus-config.yaml
kubectl apply -f $MONITORING_DIR/prometheus-deployment.yaml
kubectl rollout status deployment/prometheus -n $NAMESPACE --timeout=120s
echo "✓ Prometheus deployed"

# 4. Deploy Alertmanager
echo ""
echo "4. Deploying Alertmanager..."
kubectl apply -f $MONITORING_DIR/alertmanager-config.yaml
kubectl rollout status deployment/alertmanager -n $NAMESPACE --timeout=60s
echo "✓ Alertmanager deployed"

# 5. Deploy Grafana
echo ""
echo "5. Deploying Grafana..."
kubectl apply -f $MONITORING_DIR/grafana-config.yaml
kubectl apply -f $MONITORING_DIR/grafana-deployment.yaml
kubectl rollout status deployment/grafana -n $NAMESPACE --timeout=120s
echo "✓ Grafana deployed"

# 6. Deploy Loki + Promtail
echo ""
echo "6. Deploying Loki + Promtail..."
kubectl apply -f $MONITORING_DIR/loki-logging.yaml
kubectl rollout status deployment/loki -n $NAMESPACE --timeout=120s
echo "✓ Loki and Promtail deployed"

# 7. Verify everything
echo ""
echo "7. Verifying all components..."
echo ""
kubectl get pods -n $NAMESPACE | grep -E "prometheus|grafana|loki|alertmanager|promtail"

# 8. Print access instructions
echo ""
echo "=========================================="
echo "✅ Complete monitoring stack deployed!"
echo "=========================================="
echo ""
echo "📊 Access monitoring tools:"
echo ""
echo "  Prometheus:"
echo "    kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE &"
echo "    Open: http://localhost:9090"
echo ""
echo "  Grafana:"
echo "    kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE &"
echo "    Open: http://localhost:3000"
echo "    Login: admin / admin"
echo ""
echo "  Alertmanager:"
echo "    kubectl port-forward svc/alertmanager 9093:9093 -n $NAMESPACE &"
echo "    Open: http://localhost:9093"
echo ""
echo "📋 Useful commands:"
echo ""
echo "  Check monitoring pods:"
echo "    kubectl get pods -n $NAMESPACE"
echo ""
echo "  View logs:"
echo "    kubectl logs -f deployment/prometheus -n $NAMESPACE"
echo "    kubectl logs -f deployment/grafana -n $NAMESPACE"
echo "    kubectl logs -f deployment/loki -n $NAMESPACE"
echo ""
echo "  Get events:"
echo "    kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'"
echo ""
echo "  Check alerts in Prometheus:"
echo "    kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE &"
echo "    Visit http://localhost:9090/alerts"
echo ""
echo "🔐 Slack Integration:"
if [ -n "$SLACK_WEBHOOK" ]; then
    echo "    ✓ Webhook configured"
    echo "    Alerts will be sent to Slack"
else
    echo "    ⚠️  Not configured"
    echo "    To setup later:"
    echo "    kubectl create secret generic alertmanager-secrets -n $NAMESPACE \\"
    echo "      --from-literal=slack_webhook_url='https://hooks.slack.com/services/...'"
    echo "    kubectl rollout restart deployment/alertmanager -n $NAMESPACE"
fi
echo ""
echo "📝 Next steps:"
echo "  1. Port-forward to Grafana and configure data sources"
echo "  2. Import dashboards for Model Performance, Infrastructure, Data Drift"
echo "  3. Setup Slack integration if not done"
echo "  4. Configure Prometheus to scrape API metrics (/metrics endpoint)"
echo "  5. Test alert firing to verify Slack integration"
echo ""
