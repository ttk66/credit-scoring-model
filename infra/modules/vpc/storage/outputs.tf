output "model_bucket_name" {
  description = "Name of the bucket for ML models"
  value       = yandex_storage_bucket.models.bucket
}

output "data_bucket_name" {
  description = "Name of the bucket for data"
  value       = yandex_storage_bucket.data.bucket
}

output "config_bucket_name" {
  description = "Name of the bucket for configurations"
  value       = yandex_storage_bucket.configs.bucket
}

output "storage_access_key" {
  description = "Access key for storage"
  value       = yandex_iam_service_account_static_access_key.bucket_access.access_key
  sensitive   = true
}

output "storage_secret_key" {
  description = "Secret key for storage"
  value       = yandex_iam_service_account_static_access_key.bucket_access.secret_key
  sensitive   = true
}

output "storage_service_account_id" {
  description = "ID of the storage service account"
  value       = yandex_iam_service_account.storage_user.id
}