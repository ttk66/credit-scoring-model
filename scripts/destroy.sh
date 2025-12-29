#!/bin/bash

set -e

ENVIRONMENT=${1:-dev}

echo "Destroying infrastructure for environment: $ENVIRONMENT"

read -p "Are you sure you want to destroy ALL infrastructure? (yes/no): " confirm
if [ "$confirm" = "yes" ]; then
  terraform destroy -var-file="terraform.tfvars" -var="environment=$ENVIRONMENT" -auto-approve
  echo "Infrastructure destroyed successfully!"
else
  echo "Destruction cancelled."
fi