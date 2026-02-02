variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "tags" {
  description = "Tags for resources"
  type        = map(string)
}

variable "container_registry_id" {
  description = "Yandex Container Registry ID"
  type        = string
  default     = "crpn3tq7q9d6m8i8e5vn"
}

variable "storage_access_key" {
  description = "S3/Storage access key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "storage_secret_key" {
  description = "S3/Storage secret key"
  type        = string
  sensitive   = true
  default     = ""
}

variable "model_bucket_name" {
  description = "Name of the bucket for ML models"
  type        = string
  default     = "credit-scoring-models"
}

variable "api_domain" {
  description = "API domain name"
  type        = string
  default     = "api.credit-scoring.example.com"
}

variable "replicas" {
  description = "Number of replicas for API deployment"
  type        = number
  default     = 2
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}
