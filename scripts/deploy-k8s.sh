#!/bin/bash
set -e

# Configuration
NAMESPACE="ml-serving"
ENVIRONMENT=${1:-staging}
KUBECONFIG=${2:-~/.kube/config}

echo "Deploying to ${ENVIRONMENT} environment..."

# Apply namespace
kubectl apply -f kubernetes/namespaces/ml-serving.yaml

# Apply ConfigMaps
echo "Applying ConfigMaps..."
kubectl apply -f kubernetes/configs/ -n ${NAMESPACE}

# Apply Secrets
echo "Applying Secrets..."
sops -d kubernetes/secrets/storage-secret.enc.yaml | kubectl apply -f - -n ${NAMESPACE}
sops -d kubernetes/secrets/database-secret.enc.yaml | kubectl apply -f - -n ${NAMESPACE}

# Apply Storage
echo "Applying Storage..."
kubectl apply -f kubernetes/storage/ -n ${NAMESPACE}

# Apply Deployments with rolling update
echo "Applying Deployments..."
for deployment in kubernetes/deployments/*.yaml; do
    echo "Deploying $(basename $deployment)"
    kubectl apply -f ${deployment} -n ${NAMESPACE}
    
    #   rollout
    DEPLOYMENT_NAME=$(yq eval '.metadata.name' ${deployment})
    kubectl rollout status deployment/${DEPLOYMENT_NAME} -n ${NAMESPACE} --timeout=300s
done

# Apply Services
echo "Applying Services..."
kubectl apply -f kubernetes/services/ -n ${NAMESPACE}

# Apply Ingress
echo "Applying Ingress..."
kubectl apply -f kubernetes/ingress/ -n ${NAMESPACE}

# Apply Autoscaling
echo "Applying Autoscaling..."
kubectl apply -f kubernetes/autoscaling/ -n ${NAMESPACE}

#  
echo "Checking deployment status..."
kubectl get all -n ${NAMESPACE}
kubectl get ingress -n ${NAMESPACE}
kubectl get hpa -n ${NAMESPACE}

echo "Deployment completed successfully!"