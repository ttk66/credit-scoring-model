#!/bin/bash

set -e

ENVIRONMENT=${1:-dev}

echo "Deploying infrastructure for environment: $ENVIRONMENT"

terraform init -reconfigure \
  -backend-config="bucket=credit-scoring-terraform-state" \
  -backend-config="access_key=$YC_ACCESS_KEY" \
  -backend-config="secret_key=$YC_SECRET_KEY"

terraform validate

terraform plan -var-file="terraform.tfvars" -var="environment=$ENVIRONMENT"
read -p "Apply changes? (yes/no): " confirm
if [ "$confirm" = "yes" ]; then
  terraform apply -var-file="terraform.tfvars" -var="environment=$ENVIRONMENT" -auto-approve
  echo "Infrastructure deployed successfully!"
  
  terraform output -json > outputs.json
  echo "Outputs saved to outputs.json"
else
  echo "Deployment cancelled."
fi