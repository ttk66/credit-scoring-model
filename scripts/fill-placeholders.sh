#!/bin/bash

# Скрипт для интерактивного заполнения всех заглушек в конфигурации
# Использование: bash fill-placeholders.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================="
echo "Yandex Cloud Configuration Helper"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if yc CLI is installed
if ! command -v yc &> /dev/null; then
    print_error "Yandex Cloud CLI (yc) is not installed"
    echo "Install from: https://cloud.yandex.ru/docs/cli/quickstart"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    print_error "jq is not installed (required to parse yc JSON output)"
    echo "Install jq (e.g., apt install jq, yum install jq, or see https://stedolan.github.io/jq/)"
    exit 1
fi

print_step "Step 1: Getting Yandex Cloud credentials from YC CLI"
echo ""

# Get current config
YC_CONFIG=$(yc config list 2>/dev/null || true)

# Extract values
YC_CLOUD_ID=$(echo "$YC_CONFIG" | grep "cloud-id:" | awk '{print $2}' || echo "")
YC_FOLDER_ID=$(echo "$YC_CONFIG" | grep "folder-id:" | awk '{print $2}' || echo "")

if [ -z "$YC_CLOUD_ID" ]; then
    print_info "Cloud ID not found in yc config. Run: yc init"
    read -p "Enter Yandex Cloud ID: " YC_CLOUD_ID
fi

if [ -z "$YC_FOLDER_ID" ]; then
    print_info "Folder ID not found in yc config. Run: yc init"
    read -p "Enter Yandex Cloud Folder ID: " YC_FOLDER_ID
fi

print_info "Cloud ID: $YC_CLOUD_ID"
print_info "Folder ID: $YC_FOLDER_ID"
echo ""

# Get Zone
print_step "Step 2: Select Yandex Cloud Zone"
read -p "Enter zone (default: ru-central1-a): " YC_ZONE
YC_ZONE=${YC_ZONE:-ru-central1-a}
print_info "Zone: $YC_ZONE"
echo ""

# Get or create OAuth token
print_step "Step 3: Getting OAuth Token"
print_info "If you don't have a token, get one here:"
echo "https://oauth.yandex.ru/authorize?response_type=token&client_id=1a6990511fc648e8a709&redirect_uri=https://oauth.yandex.ru/verification_code"
echo ""
read -s -p "Paste your OAuth token: " YC_TOKEN
echo ""
echo ""

# Get or create Object Storage credentials
print_step "Step 4: Getting Object Storage Access Key"
print_info "Creating/getting Service Account for Terraform..."

SA_NAME="terraform-sa"
print_info "Checking if Service Account '$SA_NAME' exists..."

SA_ID=$(yc iam service-account get "$SA_NAME" 2>/dev/null | grep "^id:" | awk '{print $2}' || echo "")

if [ -z "$SA_ID" ]; then
    print_info "Creating new Service Account..."
    yc iam service-account create --name "$SA_NAME" --folder-id "$YC_FOLDER_ID"
    SA_ID=$(yc iam service-account get "$SA_NAME" --folder-id "$YC_FOLDER_ID" | grep "^id:" | awk '{print $2}')
    print_info "Service Account created: $SA_ID"
else
    print_info "Service Account found: $SA_ID"
fi

# Create static access key
print_info "Creating static access key..."
KEY_FILE="/tmp/yc-key-$$.json"

yc iam access-key create \
    --service-account-id "$SA_ID" \
    --format json > "$KEY_FILE"

# Parse access key and secret with fallbacks for different yc CLI output shapes
YC_ACCESS_KEY=$(jq -r '
    if (.access_key and .access_key.key_id) then .access_key.key_id
    elif (.access_key and .access_key.id) then .access_key.id
    elif .id then .id
    else empty end' "$KEY_FILE" 2>/dev/null || true)

YC_SECRET_KEY=$(jq -r '
    if (.secret and (.secret | type) == "object" and .secret.secret_key) then .secret.secret_key
    elif (.secret and (.secret | type) == "string") then .secret
    elif .secret_key then .secret_key
    elif .value then .value
    else empty end' "$KEY_FILE" 2>/dev/null || true)

if [ -z "$YC_ACCESS_KEY" ] || [ -z "$YC_SECRET_KEY" ]; then
        print_error "Failed to parse access key JSON from yc. Showing raw output for debugging:"
        echo "--- BEGIN $KEY_FILE ---"
        sed -n '1,200p' "$KEY_FILE" || true
        echo "--- END $KEY_FILE ---"
        print_error "Please check the yc CLI version and ensure 'yc iam access-key create --format json' returns the expected JSON structure."
        exit 1
fi

print_info "Access Key: $YC_ACCESS_KEY"
print_info "Secret Key: [HIDDEN]"
echo ""

# Get domain names
print_step "Step 5: Domain Configuration"
print_info "You'll need to set up domain names for Ingress (with SSL/TLS)"
print_info "Options:"
echo "  1. Use an existing domain"
echo "  2. Use IP address (for dev/testing)"
echo ""

read -p "Do you have a domain name? (y/n): " HAS_DOMAIN

if [[ $HAS_DOMAIN =~ ^[Yy]$ ]]; then
    read -p "Enter API domain (e.g., api.yourdomain.com): " API_DOMAIN
    read -p "Enter App domain (e.g., app.yourdomain.com): " APP_DOMAIN
else
    print_info "You can assign a LoadBalancer IP later after deploying K8s"
    API_DOMAIN="api.credit-scoring.example.com"
    APP_DOMAIN="app.credit-scoring.example.com"
    print_info "Using placeholder domains (update after deployment)"
fi

print_info "API Domain: $API_DOMAIN"
print_info "App Domain: $APP_DOMAIN"
echo ""

# Email for cert-manager
print_step "Step 6: SSL Certificate Configuration"
read -p "Enter email for Let's Encrypt notifications: " CERT_EMAIL
print_info "Email: $CERT_EMAIL"
echo ""

# Get project name
print_step "Step 7: Project Configuration"
read -p "Enter project name (default: credit-scoring): " PROJECT_NAME
PROJECT_NAME=${PROJECT_NAME:-credit-scoring}
print_info "Project Name: $PROJECT_NAME"
echo ""

# Get environment
print_step "Step 8: Environment Selection"
echo "Available environments: dev, staging, prod"
read -p "Select environment (default: dev): " ENVIRONMENT
ENVIRONMENT=${ENVIRONMENT:-dev}
print_info "Environment: $ENVIRONMENT"
echo ""

# Generate terraform.tfvars
print_step "Creating infra/terraform.tfvars..."

cat > "$REPO_ROOT/infra/terraform.tfvars" << EOF
# Generated by fill-placeholders.sh
# $(date)

# Yandex Cloud Credentials
yc_token     = "$YC_TOKEN"
yc_cloud_id  = "$YC_CLOUD_ID"
yc_folder_id = "$YC_FOLDER_ID"
yc_zone      = "$YC_ZONE"

# Object Storage
yc_access_key = "$YC_ACCESS_KEY"
yc_secret_key = "$YC_SECRET_KEY"

# Project Configuration
project_name = "$PROJECT_NAME"
environment  = "$ENVIRONMENT"

# Kubernetes Configuration
k8s_version     = "1.33"
k8s_node_count  = 3
k8s_node_cpu    = 4
k8s_node_memory = 8
k8s_node_disk_size = 50

# VPC Configuration
vpc_cidr = "10.0.0.0/16"
EOF

print_info "infra/terraform.tfvars created"
echo ""

# Update Ingress configuration
print_step "Updating Kubernetes Ingress configuration..."

INGRESS_FILE="$REPO_ROOT/kubernetes/ingress/ingress.yaml"
if [[ $HAS_DOMAIN =~ ^[Yy]$ ]]; then
    if [ -f "$INGRESS_FILE" ]; then
        sed -i "s/api.credit-scoring.example.com/$API_DOMAIN/g" "$INGRESS_FILE" 2>/dev/null || \
        sed -i '' "s/api.credit-scoring.example.com/$API_DOMAIN/g" "$INGRESS_FILE"

        sed -i "s/app.credit-scoring.example.com/$APP_DOMAIN/g" "$INGRESS_FILE" 2>/dev/null || \
        sed -i '' "s/app.credit-scoring.example.com/$APP_DOMAIN/g" "$INGRESS_FILE"

        print_info "✓ Ingress domains updated"
    else
        print_info "Ingress file not found at $INGRESS_FILE; skipping domain replacement"
    fi
fi

echo ""

# Update cert-manager configuration
print_step "Updating Cert-Manager configuration..."

CERT_FILE="$REPO_ROOT/kubernetes/cert-manager/cert-manager-install.yaml"
if [ -f "$CERT_FILE" ]; then
    sed -i "s/admin@credit-scoring.example.com/$CERT_EMAIL/g" "$CERT_FILE" 2>/dev/null || \
    sed -i '' "s/admin@credit-scoring.example.com/$CERT_EMAIL/g" "$CERT_FILE"

    print_info "Cert-Manager email updated"
else
    print_info "Cert-Manager file not found at $CERT_FILE; skipping email replacement"
fi
echo ""

# Save keys to secure location
print_step "Saving credentials to secure location..."

SECRETS_DIR="$REPO_ROOT/.secrets"
mkdir -p "$SECRETS_DIR"
chmod 700 "$SECRETS_DIR"

cp "$KEY_FILE" "$SECRETS_DIR/terraform-key.json"

print_info "Credentials saved to .secrets/"
echo ""

# Add to gitignore
if ! grep -q "infra/terraform.tfvars" "$REPO_ROOT/.gitignore" 2>/dev/null; then
    echo "infra/terraform.tfvars" >> "$REPO_ROOT/.gitignore"
fi

if ! grep -q "^terraform.tfvars$" "$REPO_ROOT/.gitignore" 2>/dev/null; then
    echo "terraform.tfvars" >> "$REPO_ROOT/.gitignore"
fi

if ! grep -q ".secrets/" "$REPO_ROOT/.gitignore" 2>/dev/null; then
    echo ".secrets/" >> "$REPO_ROOT/.gitignore"
fi

print_info "Added to .gitignore"
echo ""

# Summary
echo "=================================="
echo -e "${GREEN}Configuration Complete!${NC}"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Review infra/terraform.tfvars (do NOT commit this file!)"
echo "2. Initialize Terraform:"
echo "   cd $REPO_ROOT/infra"
echo "   terraform init -reconfigure \\"
echo "     -backend-config='access_key=$YC_ACCESS_KEY' \\"
echo "     -backend-config='secret_key=$YC_SECRET_KEY'"
echo ""
echo "3. Validate configuration:"
echo "   terraform validate"
echo ""
echo "4. Plan deployment:"
echo "   terraform plan"
echo ""
echo "5. Apply configuration:"
echo "   terraform apply"
echo ""
echo "Important:"
echo "  Never commit terraform.tfvars to git"
echo "  Keep .secrets/ directory secure"
echo "  Rotate credentials periodically"
echo ""

# Cleanup
rm -f "$KEY_FILE"

print_info "Done!"
