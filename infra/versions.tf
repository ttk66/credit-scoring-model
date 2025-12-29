terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.95.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.23.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.11.0"
    }
  }

  # Настройка remote state в Yandex Object Storage
  backend "s3" {
    endpoint   = "storage.yandexcloud.net"
    bucket     = "credit-scoring-terraform-state"
    region     = "ru-central1"
    key        = "terraform.tfstate"
    access_key = ""  # Заполняется через переменные
    secret_key = ""  # Заполняется через переменные

    skip_region_validation      = true
    skip_credentials_validation = true
    skip_metadata_api_check     = true
  }
}