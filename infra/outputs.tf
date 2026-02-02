output "k8s_cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = module.kubernetes.cluster_external_v4_endpoint
}

output "k8s_cluster_ca_certificate" {
  description = "Kubernetes cluster CA certificate"
  value       = module.kubernetes.cluster_ca_certificate
  sensitive   = true
}

output "model_bucket_name" {
  description = "Name of the bucket for ML models"
  value       = "credit-scoring-dev-models"  # TODO: создать через yc CLI
}

# TODO: Мониторинг будет добавлен позже
# output "monitoring_dashboard_url" {
#   description = "URL of the monitoring dashboard"
#   value       = module.monitoring.dashboard_url
# }

# TODO: ML serving outputs будут добавлены позже
# output "ml_api_endpoint" {
#   description = "Endpoint for ML model API"
#   value       = module.ml_serving.api_endpoint
# }

output "vpc_id" {
  description = "ID of the created VPC"
  value       = module.vpc.vpc_id
}

output "subnet_ids" {
  description = "IDs of the created subnets"
  value       = module.vpc.subnet_ids
}

output "service_account_key" {
  description = "Service account static access key"
  value       = yandex_iam_service_account_static_access_key.terraform_key.access_key
  sensitive   = true
}