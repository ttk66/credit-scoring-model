#!/bin/bash

#     placeholder'  kubernetes manifests
#  docker files

set -e

echo ""
echo "        Placeholder' - Kubernetes Manifest       "
echo ""
echo ""

# ============================================
# 
# ============================================

echo "===  ==="

#       
REGISTRY=${REGISTRY:-"cr.yandex/crpn3tq7q9d6m8i8e5vn"}
CLUSTER_NAME=${CLUSTER_NAME:-"credit-scoring-cluster"}
PROJECT_NAME=${PROJECT_NAME:-"credit-scoring-model"}
DOMAIN=${DOMAIN:-"credit-scoring.example.com"}
ENVIRONMENT=${ENVIRONMENT:-"dev"}

echo "Registry: $REGISTRY"
echo "Cluster: $CLUSTER_NAME"
echo "Project: $PROJECT_NAME"
echo "Domain: $DOMAIN"
echo "Environment: $ENVIRONMENT"
echo ""

#     
read -p "   ? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    read -p " Registry: " REGISTRY
    read -p " Cluster Name: " CLUSTER_NAME
    read -p " Project Name: " PROJECT_NAME
    read -p " Domain: " DOMAIN
    read -p " Environment: " ENVIRONMENT
fi

# ============================================
# 
# ============================================

echo ""
echo "===   ==="

#    placeholder'
replace_placeholder() {
    local file=$1
    local placeholder=$2
    local value=$3
    
    if grep -q "$placeholder" "$file"; then
        sed -i "s|$placeholder|$value|g" "$file"
        echo "$file: $placeholder  $value"
    fi
}

#      
replace_in_all() {
    local placeholder=$1
    local value=$2
    local pattern=$3
    
    echo ""
    echo ": $placeholder  $value"
    find . -name "$pattern" -type f | while read file; do
        replace_placeholder "$file" "$placeholder" "$value"
    done
}

# ============================================
# 
# ============================================

# Registry
replace_in_all "cr.yandex/YOUR_REGISTRY_ID" "$REGISTRY" "*.yaml"
replace_in_all "YOUR_REGISTRY" "$REGISTRY" "*.yaml"

# Cluster Name
replace_in_all "credit-scoring-cluster" "$CLUSTER_NAME" "*.yaml"

# Namespace
replace_in_all "ml-serving" "ml-serving" "*.yaml"

# Domain
replace_in_all "credit-scoring.example.com" "$DOMAIN" "*.yaml"
replace_in_all "api.credit-scoring.example.com" "api.$DOMAIN" "*.yaml"
replace_in_all "app.credit-scoring.example.com" "app.$DOMAIN" "*.yaml"

# Environment
replace_in_all "ENVIRONMENT: production" "ENVIRONMENT: $ENVIRONMENT" "*.yaml"

# Database
DB_USER=${DB_USER:-"postgres"}
DB_PASSWORD=${DB_PASSWORD:-"$(openssl rand -base64 32)"}
DB_NAME=${DB_NAME:-"credit_scoring"}
DB_HOST=${DB_HOST:-"postgresql.ml-serving.svc.cluster.local"}

replace_in_all "postgres" "$DB_USER" "all-configmaps.yaml"
replace_in_all "credit_scoring" "$DB_NAME" "all-configmaps.yaml"
replace_in_all "postgresql.ml-serving.svc.cluster.local" "$DB_HOST" "all-configmaps.yaml"

# Redis
REDIS_PASSWORD=${REDIS_PASSWORD:-"$(openssl rand -base64 32)"}
replace_in_all "redis-password-placeholder" "$REDIS_PASSWORD" "all-configmaps.yaml"

# Storage/DVC
S3_ACCESS_KEY=${S3_ACCESS_KEY:-"your-access-key"}
S3_SECRET_KEY=${S3_SECRET_KEY:-"your-secret-key"}
S3_ENDPOINT=${S3_ENDPOINT:-"https://storage.yandexcloud.net"}
S3_BUCKET=${S3_BUCKET:-"credit-scoring-models"}

replace_in_all "your-access-key" "$S3_ACCESS_KEY" "all-configmaps.yaml"
replace_in_all "your-secret-key" "$S3_SECRET_KEY" "all-configmaps.yaml"
replace_in_all "https://storage.yandexcloud.net" "$S3_ENDPOINT" "all-configmaps.yaml"

# Docker Registry
DOCKER_SERVER=${DOCKER_SERVER:-"cr.yandex"}
DOCKER_USERNAME=${DOCKER_USERNAME:-"oauth"}
DOCKER_PASSWORD=${DOCKER_PASSWORD:-"your-token"}

replace_in_all "docker-server-placeholder" "$DOCKER_SERVER" "all-configmaps.yaml"
replace_in_all "docker-username-placeholder" "$DOCKER_USERNAME" "all-configmaps.yaml"
replace_in_all "docker-password-placeholder" "$DOCKER_PASSWORD" "all-configmaps.yaml"

# API Configuration
API_PORT=${API_PORT:-"8000"}
API_WORKERS=${API_WORKERS:-"4"}
API_LOG_LEVEL=${API_LOG_LEVEL:-"INFO"}

replace_in_all "8000" "$API_PORT" "api-deployment.yaml"
replace_in_all "API_WORKERS: 4" "API_WORKERS: $API_WORKERS" "api-deployment.yaml"
replace_in_all "LOG_LEVEL: INFO" "LOG_LEVEL: $API_LOG_LEVEL" "api-deployment.yaml"

# Limits and Requests
API_CPU_REQUEST=${API_CPU_REQUEST:-"250m"}
API_CPU_LIMIT=${API_CPU_LIMIT:-"1000m"}
API_MEMORY_REQUEST=${API_MEMORY_REQUEST:-"512Mi"}
API_MEMORY_LIMIT=${API_MEMORY_LIMIT:-"2Gi"}

replace_in_all "cpu: 250m" "cpu: $API_CPU_REQUEST" "api-deployment.yaml"
replace_in_all "cpu: 1000m" "cpu: $API_CPU_LIMIT" "api-deployment.yaml"
replace_in_all "memory: 512Mi" "memory: $API_MEMORY_REQUEST" "api-deployment.yaml"
replace_in_all "memory: 2Gi" "memory: $API_MEMORY_LIMIT" "api-deployment.yaml"

echo ""
echo " placeholder' !"

# ============================================
#  
# ============================================

echo ""
echo "                                                               "
echo "                                         "
echo ""
echo ""
echo "Registry:              $REGISTRY"
echo "Cluster:              $CLUSTER_NAME"
echo "Project:              $PROJECT_NAME"
echo "Domain:               $DOMAIN"
echo "Environment:          $ENVIRONMENT"
echo ""
echo "Database:"
echo "  User:               $DB_USER"
echo "  Host:               $DB_HOST"
echo "  Database:           $DB_NAME"
echo "  Password:           $(echo $DB_PASSWORD | head -c 10)..."
echo ""
echo "Redis:"
echo "  Password:           $(echo $REDIS_PASSWORD | head -c 10)..."
echo ""
echo "S3/Storage:"
echo "  Endpoint:           $S3_ENDPOINT"
echo "  Bucket:             $S3_BUCKET"
echo "  Access Key:         $(echo $S3_ACCESS_KEY | head -c 10)..."
echo ""
echo "Docker:"
echo "  Server:             $DOCKER_SERVER"
echo "  Username:           $DOCKER_USERNAME"
echo ""
echo "API:"
echo "  Port:               $API_PORT"
echo "  Workers:            $API_WORKERS"
echo "  Log Level:          $API_LOG_LEVEL"
echo "  CPU Request:        $API_CPU_REQUEST"
echo "  CPU Limit:          $API_CPU_LIMIT"
echo "  Memory Request:     $API_MEMORY_REQUEST"
echo "  Memory Limit:       $API_MEMORY_LIMIT"
echo ""

# ============================================
#  
# ============================================

echo " ..."
cat > .env.deployed << EOF
REGISTRY=$REGISTRY
CLUSTER_NAME=$CLUSTER_NAME
PROJECT_NAME=$PROJECT_NAME
DOMAIN=$DOMAIN
ENVIRONMENT=$ENVIRONMENT
DB_USER=$DB_USER
DB_PASSWORD=$DB_PASSWORD
DB_NAME=$DB_NAME
DB_HOST=$DB_HOST
REDIS_PASSWORD=$REDIS_PASSWORD
S3_ACCESS_KEY=$S3_ACCESS_KEY
S3_SECRET_KEY=$S3_SECRET_KEY
S3_ENDPOINT=$S3_ENDPOINT
S3_BUCKET=$S3_BUCKET
DOCKER_SERVER=$DOCKER_SERVER
DOCKER_USERNAME=$DOCKER_USERNAME
DOCKER_PASSWORD=$DOCKER_PASSWORD
API_PORT=$API_PORT
API_WORKERS=$API_WORKERS
API_LOG_LEVEL=$API_LOG_LEVEL
API_CPU_REQUEST=$API_CPU_REQUEST
API_CPU_LIMIT=$API_CPU_LIMIT
API_MEMORY_REQUEST=$API_MEMORY_REQUEST
API_MEMORY_LIMIT=$API_MEMORY_LIMIT
EOF

echo "   .env.deployed"
echo ""

# Attempt to replace remaining placeholders in kubernetes/configs/all-configmaps.yaml
CONFIGMAP_FILE="kubernetes/configs/all-configmaps.yaml"
if [ -f "$CONFIGMAP_FILE" ]; then
    if grep -q "YOUR_\|YOUR_BASE64_ENCODED_CREDENTIALS_HERE" "$CONFIGMAP_FILE"; then
        echo "  placeholder'  $CONFIGMAP_FILE.  ." 

        if [ -f .env.deployed ]; then
            export $(grep -v '^#' .env.deployed | xargs)
        fi

        read -p "S3 access key (leave empty to use S3_ACCESS_KEY from .env.deployed): " S3_ACCESS_KEY_INPUT
        S3_ACCESS_KEY=${S3_ACCESS_KEY_INPUT:-${S3_ACCESS_KEY:-}}
        read -p "S3 secret key (leave empty to use S3_SECRET_KEY from .env.deployed): " S3_SECRET_KEY_INPUT
        S3_SECRET_KEY=${S3_SECRET_KEY_INPUT:-${S3_SECRET_KEY:-}}

        read -p "Docker username (default: $DOCKER_USERNAME): " DOCKER_USERNAME_INPUT
        DOCKER_USERNAME=${DOCKER_USERNAME_INPUT:-${DOCKER_USERNAME:-oauth}}
        read -s -p "Docker password (leave empty to use DOCKER_PASSWORD from .env.deployed): " DOCKER_PASSWORD_INPUT
        echo
        DOCKER_PASSWORD=${DOCKER_PASSWORD_INPUT:-${DOCKER_PASSWORD:-}}

        if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ]; then
            if command -v base64 >/dev/null 2>&1; then
                DOCKER_AUTH=$(printf "%s:%s" "$DOCKER_USERNAME" "$DOCKER_PASSWORD" | base64 | tr -d '\n')
            else
                DOCKER_AUTH=""
            fi
        else
            DOCKER_AUTH=""
        fi

        if [ -n "$DOCKER_AUTH" ]; then
            sed -i "s|YOUR_BASE64_ENCODED_CREDENTIALS_HERE|$DOCKER_AUTH|g" "$CONFIGMAP_FILE" 2>/dev/null || \
            sed -i '' "s|YOUR_BASE64_ENCODED_CREDENTIALS_HERE|$DOCKER_AUTH|g" "$CONFIGMAP_FILE" || true
        fi

        if [ -n "$S3_ACCESS_KEY" ]; then
            sed -i "s|YOUR_YANDEX_ACCESS_KEY|$S3_ACCESS_KEY|g" "$CONFIGMAP_FILE" 2>/dev/null || \
            sed -i '' "s|YOUR_YANDEX_ACCESS_KEY|$S3_ACCESS_KEY|g" "$CONFIGMAP_FILE" || true
            sed -i "s|YOUR_DVC_ACCESS_KEY|$S3_ACCESS_KEY|g" "$CONFIGMAP_FILE" 2>/dev/null || \
            sed -i '' "s|YOUR_DVC_ACCESS_KEY|$S3_ACCESS_KEY|g" "$CONFIGMAP_FILE" || true
        fi

        if [ -n "$S3_SECRET_KEY" ]; then
            sed -i "s|YOUR_YANDEX_SECRET_KEY|$S3_SECRET_KEY|g" "$CONFIGMAP_FILE" 2>/dev/null || \
            sed -i '' "s|YOUR_YANDEX_SECRET_KEY|$S3_SECRET_KEY|g" "$CONFIGMAP_FILE" || true
            sed -i "s|YOUR_DVC_SECRET_KEY|$S3_SECRET_KEY|g" "$CONFIGMAP_FILE" 2>/dev/null || \
            sed -i '' "s|YOUR_DVC_SECRET_KEY|$S3_SECRET_KEY|g" "$CONFIGMAP_FILE" || true
        fi

        echo "   $CONFIGMAP_FILE    placeholder':" 
        grep -n "YOUR_\|YOUR_BASE64_ENCODED_CREDENTIALS_HERE" "$CONFIGMAP_FILE" || echo "OK"
    fi
fi

# ============================================
#  
# ============================================

echo ""
echo "                                                  "
echo ""
echo ""
echo "1.    placeholder' :"
echo "   grep -r 'placeholder\\|YOUR_\\|CHANGE_ME' kubernetes/"
echo ""
echo "2.  Docker Registry Secret:"
echo "   kubectl create secret docker-registry docker-registry-credentials \\"
echo "     --docker-server=$DOCKER_SERVER \\"
echo "     --docker-username=$DOCKER_USERNAME \\"
echo "     --docker-password='$DOCKER_PASSWORD' \\"
echo "     -n ml-serving"
echo ""
echo "3.  Database Secret:"
echo "   kubectl create secret generic database-credentials \\"
echo "     --from-literal=username=$DB_USER \\"
echo "     --from-literal=password='$DB_PASSWORD' \\"
echo "     -n ml-serving"
echo ""
echo "4.   :"
echo "   bash scripts/full-deployment.sh"
echo ""
echo "5.  :"
echo "   kubectl get pods -n ml-serving"
echo "   kubectl get svc -n ml-serving"
echo ""
