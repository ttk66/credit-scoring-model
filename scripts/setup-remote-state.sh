#!/bin/bash

set -e

echo "Setting up Terraform remote state..."

yc storage bucket create \
  --name credit-scoring-terraform-state \
  --default-storage-class standard \
  --max-size 1073741824  # 1GB

yc iam service-account create --name terraform-sa

yc resource-manager folder add-access-binding \
  --role editor \
  --subject serviceAccount:terraform-sa

yc resource-manager folder add-access-binding \
  --role storage.editor \
  --subject serviceAccount:terraform-sa

yc iam service-account create-access-key --name terraform-sa > terraform-key.json

echo "Remote state setup complete!"
echo "Please add the access key to your terraform.tfvars file:"
cat terraform-key.json