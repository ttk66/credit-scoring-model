#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Variables
REGISTRY="cr.yandex/crpn3tq7q9d6m8i8e5vn"
TAG=${1:-latest}
ENVIRONMENT=${2:-production}

echo -e "${GREEN}Building and pushing Docker images...${NC}"
echo -e "Registry: ${REGISTRY}"
echo -e "Tag: ${TAG}"
echo -e "Environment: ${ENVIRONMENT}"

#      
build_and_push() {
    local service=$1
    local context=$2
    local dockerfile=$3
    
    echo -e "\n${YELLOW}Building ${service}...${NC}"
    
    #  
    docker build \
        -t "${REGISTRY}/${service}:${TAG}" \
        -t "${REGISTRY}/${service}:${ENVIRONMENT}" \
        -f "${dockerfile}" \
        "${context}"
    
    #   
    echo -e "Image size:"
    docker images "${REGISTRY}/${service}:${TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
    # Push 
    echo -e "${YELLOW}Pushing ${service}...${NC}"
    docker push "${REGISTRY}/${service}:${TAG}"
    docker push "${REGISTRY}/${service}:${ENVIRONMENT}"
    
    echo -e "${GREEN} ${service} built and pushed successfully${NC}"
}

#   
build_and_push "credit-scoring-api" "docker/api" "docker/api/Dockerfile"
build_and_push "credit-scoring-frontend" "docker/frontend" "docker/frontend/Dockerfile"
build_and_push "credit-scoring-data-loader" "docker/data-loader" "docker/data-loader/Dockerfile"

# Trivy   
echo -e "\n${YELLOW}Scanning for vulnerabilities...${NC}"
for service in credit-scoring-api credit-scoring-frontend credit-scoring-data-loader; do
    echo -e "Scanning ${service}..."
    docker run --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        aquasec/trivy:latest \
        image --severity HIGH,CRITICAL \
        "${REGISTRY}/${service}:${TAG}"
done

#   SBOM
echo -e "\n${YELLOW}Generating SBOM reports...${NC}"
for service in credit-scoring-api credit-scoring-frontend credit-scoring-data-loader; do
    docker run --rm \
        -v /var/run/docker.sock:/var/run/docker.sock \
        anchore/syft:latest \
        "${REGISTRY}/${service}:${TAG}" \
        -o spdx-json > "sbom-${service}-${TAG}.json"
    echo -e "SBOM generated: sbom-${service}-${TAG}.json"
done

echo -e "\n${GREEN}All images built and pushed successfully!${NC}"