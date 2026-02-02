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

  # Локальный backend для разработки
  # Для production используйте remote backend (S3 в Yandex Object Storage)
  # backend "s3" {
  #   endpoint   = "https://storage.yandexcloud.net"
  #   bucket     = "credit-scoring-terraform-state"
  #   region     = "ru-central1"
  #   key        = "terraform.tfstate"
  #   access_key = ""  # Передать через: terraform init -backend-config="access_key=..."
  #   secret_key = ""  # Передать через: terraform init -backend-config="secret_key=..."
  #
  #   skip_region_validation      = true
  #   skip_credentials_validation = true
  #   skip_metadata_api_check     = true
  # }
}