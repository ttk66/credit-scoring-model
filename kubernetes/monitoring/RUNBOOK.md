# Credit Scoring Model - Incident Runbook

## Overview
This runbook provides procedures for responding to common incidents affecting the Credit Scoring ML Model service.

## Critical Metrics Dashboard
- **Prometheus URL**: http://prometheus:9090
- **Grafana URL**: http://grafana:3000 (admin/admin)
- **Loki Logs**: http://loki:3100

---

## INCIDENT: API Service Down

### Detection
- Alert: `APIDown` (Prometheus)
- Symptom: `curl http://api-service/health` returns 503 or connection refused

### Initial Response (0-5 min)
1. Check pod status:
   ```bash
   kubectl get pods -n ml-serving | grep api
   kubectl describe pod api-<POD_ID> -n ml-serving
   ```

2. Check logs:
   ```bash
   kubectl logs -n ml-serving --tail=100 api-<POD_ID>
   ```

3. Check resource constraints:
   ```bash
   kubectl top pods -n ml-serving
   kubectl top nodes
   ```

### Mitigation Actions
- **If OOMKilled**: Increase memory limit in Kubernetes deployment, redeploy
- **If CrashLoopBackOff**: Check logs for startup errors, fix config, redeploy
- **If pending**: Check node availability, add nodes if needed

### Resolution
```bash
# Force restart
kubectl rollout restart deployment/api -n ml-serving

# Wait for rollout
kubectl rollout status deployment/api -n ml-serving

# Verify health
kubectl port-forward svc/api-service 8000:8000 -n ml-serving
curl http://localhost:8000/health
```

---

## INCIDENT: High Error Rate (Error Rate > 5%)

### Detection
- Alert: `HighErrorRate` (Prometheus)
- Metric: `rate(http_requests_total{status=~"5.."}[5m]) > 0.05`

### Initial Response (0-5 min)
1. Check error logs:
   ```bash
   kubectl logs -n ml-serving -l app=api --tail=200 | grep ERROR
   ```

2. Check recent Loki logs:
   ```bash
   # In Grafana Loki, filter: {namespace="ml-serving", container="api"}
   ```

3. Check database connectivity:
   ```bash
   kubectl exec -it api-<POD_ID> -n ml-serving -- psql -h postgres -U user -c "SELECT 1;"
   ```

### Common Root Causes & Fixes
- **Database connection pool exhausted**: Increase max_connections in PostgreSQL, restart API
- **Redis unreachable**: Check Redis pod status, restart if needed
- **Upstream service timeout**: Increase timeout in API config, redeploy
- **Model loading failure**: Check model file size, verify S3 access, redeploy

### Resolution
```bash
# Increase replicas temporarily
kubectl scale deployment/api --replicas=3 -n ml-serving

# Monitor error rate
watch kubectl top pods -n ml-serving
```

---

## INCIDENT: High Model Inference Latency (P95 > 1s)

### Detection
- Alert: `HighInferenceLatency` (Prometheus)
- Metric: `histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m])) > 1.0`

### Initial Response (0-5 min)
1. Check model server resource usage:
   ```bash
   kubectl top pods -n ml-serving | grep api
   ```

2. Check GPU availability (if applicable):
   ```bash
   kubectl get nodes -L nvidia.com/gpu
   kubectl describe node <NODE_NAME> | grep -A 5 nvidia
   ```

3. Monitor query volume:
   ```bash
   # In Prometheus: rate(model_predictions_total[1m])
   ```

### Mitigation Actions
- **If CPU constrained**: Scale horizontally (increase replicas)
- **If memory constrained**: Increase memory limit, redeploy
- **If network latency**: Check network policies, node-to-node connectivity
- **If model too large**: Consider quantization/pruning (see model_optimize scripts)

### Resolution
```bash
# Scale to handle load
kubectl scale deployment/api --replicas=5 -n ml-serving

# Enable HPA if not already active
kubectl apply -f kubernetes/autoscaling/hpa-api.yaml
```

---

## INCIDENT: Data Drift Detected

### Detection
- Alert: `DataDriftDetected` (Prometheus)
- Dashboard: "Data Drift Monitoring" in Grafana
- Metric: `data_drift_detected == 1`

### Initial Response (0-10 min)
1. Check drift metrics in Grafana:
   - Feature drift scores (by feature)
   - Target distribution changes
   - Prediction distribution

2. Pull drift report:
   ```bash
   kubectl port-forward svc/api-service 8000:8000 -n ml-serving
   curl http://localhost:8000/metrics/drift-report > drift_report.json
   ```

3. Analyze data freshness:
   ```bash
   # Check how recent training data is
   aws s3 ls s3://credit-scoring-data/raw/ | tail -5
   ```

### Mitigation Actions
- **If production drift**: Trigger model retraining via Airflow
- **If data quality issue**: Pause predictions, alert data team
- **If expected drift**: Update baseline in Evidently config

### Resolution
```bash
# Trigger retraining DAG
airflow dags trigger credit_scoring_retrain --conf '{"reason": "drift_detected"}'

# Monitor retraining job
kubectl logs -n ml-serving -f -l app=airflow-scheduler
```

---

## INCIDENT: Model Performance Degradation (AUC < 0.75)

### Detection
- Alert: `ModelPerformanceDegradation` (Prometheus)
- Metric: `model_auc < 0.75`

### Initial Response (0-15 min)
1. Collect recent predictions for analysis:
   ```bash
   # Query database for recent predictions
   kubectl exec postgres-<POD_ID> -n ml-serving -- psql -U user -c \
     "SELECT * FROM predictions WHERE created_at > NOW() - INTERVAL '1 day';"
   ```

2. Check if drift is correlated:
   ```bash
   # Cross-reference with drift metrics in Grafana
   ```

3. Review recent model version:
   ```bash
   # Check which model version is deployed
   kubectl get deploy api -n ml-serving -o jsonpath='{.spec.template.spec.containers[0].image}'
   ```

### Mitigation Actions
- **If new model caused it**: Rollback to previous model version
- **If data quality degraded**: Flag to data team, trigger manual retrain
- **If expected seasonal change**: Update model performance expectations

### Resolution
```bash
# Rollback to previous model
kubectl set image deployment/api api=<PREVIOUS_IMAGE> -n ml-serving
kubectl rollout status deployment/api -n ml-serving

# Or trigger urgent retraining
airflow dags trigger credit_scoring_retrain --conf '{"priority": "urgent"}'
```

---

## INCIDENT: Pod Restart Loop

### Detection
- Alert: `PodRestartingLoop` (Prometheus)
- Command: `kubectl get pods -n ml-serving` (watch for RESTARTS column)

### Initial Response (0-5 min)
1. Check pod events:
   ```bash
   kubectl describe pod api-<POD_ID> -n ml-serving
   ```

2. Check crash logs:
   ```bash
   kubectl logs api-<POD_ID> -n ml-serving --previous
   ```

3. Check resource limits:
   ```bash
   kubectl get deployment api -n ml-serving -o yaml | grep -A 10 resources
   ```

### Common Root Causes
- **OOMKilled**: Memory limit too low, increase it
- **Liveness probe failed**: Application hang, check logs
- **Config error**: Invalid startup config, fix and redeploy

### Resolution
```bash
# Increase memory limit
kubectl patch deployment api -n ml-serving -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"2Gi"}}}]}}}}'

# Force new rollout
kubectl rollout restart deployment/api -n ml-serving
```

---

## INCIDENT: Database Connection Issues

### Detection
- Application logs show: "psycopg2.OperationalError: could not connect to server"
- Postgres pod CrashLoopBackOff status

### Initial Response (0-5 min)
1. Check Postgres pod:
   ```bash
   kubectl get pods -n ml-serving | grep postgres
   kubectl logs postgres-<POD_ID> -n ml-serving
   ```

2. Check PVC status:
   ```bash
   kubectl get pvc -n ml-serving
   kubectl describe pvc postgres-pvc -n ml-serving
   ```

3. Test connection:
   ```bash
   kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
     psql -h postgres.ml-serving -U user -c "SELECT 1;" 2>&1
   ```

### Mitigation Actions
- **If out of disk space**: Resize PVC, restart Postgres
- **If corrupted data**: Restore from backup
- **If network issue**: Check network policy, DNS resolution

### Resolution
```bash
# Restart Postgres
kubectl delete pod postgres-<POD_ID> -n ml-serving

# Verify connection after restart
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h postgres.ml-serving -U user -c "SELECT 1;"
```

---

## INCIDENT: High Memory Usage (> 80%)

### Detection
- Alert: `HighMemoryUsage` (Prometheus)
- Metric: `container_memory_usage_bytes{pod=~"api-.*"} / container_spec_memory_limit_bytes > 0.8`

### Initial Response (0-5 min)
1. Check memory breakdown:
   ```bash
   kubectl top pods -n ml-serving
   ```

2. Identify memory hog:
   ```bash
   kubectl exec api-<POD_ID> -n ml-serving -- ps aux
   ```

3. Check for memory leaks:
   ```bash
   kubectl logs api-<POD_ID> -n ml-serving | grep -i "memory\|leak\|gc"
   ```

### Mitigation Actions
- **Temporary**: Restart pod to clear caches
- **Long-term**: Increase memory limit, optimize code

### Resolution
```bash
# Increase memory limit in deployment
kubectl patch deployment api -n ml-serving -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"2Gi"}}}]}}}}'

# Rolling restart
kubectl rollout restart deployment/api -n ml-serving
kubectl rollout status deployment/api -n ml-serving
```

---

## Emergency Escalation

If incident is not resolved within 15 minutes:

1. **Pause traffic** (if critical):
   ```bash
   kubectl scale deployment/api --replicas=0 -n ml-serving
   ```

2. **Rollback everything**:
   ```bash
   git revert HEAD  # Revert last deployment
   kubectl apply -f kubernetes/deployments/
   ```

3. **Alert team**:
   - Slack: #incidents channel
   - PagerDuty: Create P1 incident
   - Contact on-call ML engineer

4. **Post-mortem**:
   ```bash
   # Collect diagnostics
   kubectl describe nodes > /tmp/nodes.txt
   kubectl get events -n ml-serving > /tmp/events.txt
   kubectl logs -n ml-serving --all-containers=true > /tmp/logs.txt
   ```

---

## Useful Commands Reference

```bash
# Monitor in real-time
kubectl top pods -n ml-serving --sort-by=memory
kubectl top nodes

# Port-forward to local
kubectl port-forward svc/prometheus 9090:9090 -n ml-serving
kubectl port-forward svc/grafana 3000:3000 -n ml-serving
kubectl port-forward svc/loki 3100:3100 -n ml-serving

# Stream logs
kubectl logs -f deployment/api -n ml-serving
kubectl logs -f -l app=api -n ml-serving --all-containers=true

# Exec into pod
kubectl exec -it deployment/api -n ml-serving -- /bin/bash

# Check recent events
kubectl get events -n ml-serving --sort-by='.lastTimestamp'

# Describe resources
kubectl describe pod/deployment/service -n ml-serving
```

---

## See Also
- [Prometheus Alerts](./prometheus-config.yaml)
- [Grafana Dashboards](./grafana-config.yaml)
- [Loki Logging](./loki-logging.yaml)
- [Kubernetes Health Checks](../deployments/api-deployment.yaml)
