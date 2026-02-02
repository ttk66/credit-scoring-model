#!/bin/bash
set -e

NAMESPACE=${1:-ml-serving}
TIMEOUT=${2:-300}

echo "Checking rollout status for all deployments in ${NAMESPACE}..."

DEPLOYMENTS=$(kubectl get deployments -n ${NAMESPACE} -o jsonpath='{.items[*].metadata.name}')

for deployment in ${DEPLOYMENTS}; do
    echo -n "Checking ${deployment}... "
    kubectl rollout status deployment/${deployment} -n ${NAMESPACE} --timeout=${TIMEOUT}s
    if [ $? -eq 0 ]; then
        echo "${deployment} is ready"
    else
        echo "${deployment} rollout failed"
        #   
        kubectl describe deployment/${deployment} -n ${NAMESPACE}
        kubectl get pods -n ${NAMESPACE} -l app=${deployment}
        exit 1
    fi
done

#  readiness  
echo "Checking pod readiness..."
kubectl get pods -n ${NAMESPACE} -o wide

#   
echo "Checking service endpoints..."
kubectl get endpoints -n ${NAMESPACE}

echo "All deployments are ready!"