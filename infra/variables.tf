variable "yc_token" {
  description = "Yandex Cloud OAuth token"
  type        = string
  sensitive   = true
}

variable "yc_cloud_id" {
  description = "Yandex Cloud ID"
  type        = string
}

variable "yc_folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "yc_zone" {
  description = "Yandex Cloud zone"
  type        = string
  default     = "ru-central1-a"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "credit-scoring"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# Remote state variables
variable "yc_access_key" {
  description = "Yandex Cloud access key for Object Storage"
  type        = string
  sensitive   = true
}

variable "yc_secret_key" {
  description = "Yandex Cloud secret key for Object Storage"
  type        = string
  sensitive   = true
}

# VPC variables
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Kubernetes variables
variable "k8s_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.26"
}

variable "k8s_node_count" {
  description = "Number of Kubernetes nodes"
  type        = number
  default     = 3
}

variable "k8s_node_disk_size" {
  description = "Disk size for Kubernetes nodes (GB)"
  type        = number
  default     = 50
}

variable "k8s_node_cpu" {
  description = "CPU cores for Kubernetes nodes"
  type        = number
  default     = 4
}

variable "k8s_node_memory" {
  description = "Memory for Kubernetes nodes (GB)"
  type        = number
  default     = 8
}
