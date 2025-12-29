variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs"
  type        = list(string)
}

variable "k8s_version" {
  description = "Kubernetes version"
  type        = string
}

variable "node_count" {
  description = "Number of nodes"
  type        = number
}

variable "node_disk_size" {
  description = "Disk size per node (GB)"
  type        = number
}

variable "node_cpu" {
  description = "CPU cores per node"
  type        = number
}

variable "node_memory" {
  description = "Memory per node (GB)"
  type        = number
}

variable "service_account_name" {
  description = "Service account name"
  type        = string
}

variable "tags" {
  description = "Tags for resources"
  type        = map(string)
}

variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "zone" {
  description = "Zone for resources"
  type        = string
  default     = "ru-central1-a"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "security_group_id" {
  description = "Security group ID"
  type        = string
}