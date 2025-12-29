variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "k8s_cluster_id" {
  description = "Kubernetes cluster ID"
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

variable "notification_emails" {
  description = "List of email addresses for notifications"
  type        = list(string)
  default     = []
}

variable "telegram_bot_token" {
  description = "Telegram bot token for notifications"
  type        = string
  sensitive   = true
  default     = ""
}

variable "telegram_chat_id" {
  description = "Telegram chat ID for notifications"
  type        = string
  sensitive   = true
  default     = ""
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}