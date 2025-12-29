# Bucket для хранения ML моделей
resource "yandex_storage_bucket" "models" {
  bucket = "${var.name_prefix}-models"
  
  # ACL для доступа
  grant {
    type        = "Group"
    uri         = "http://acs.amazonaws.com/groups/global/AuthenticatedUsers"
    permissions = ["READ"]
  }

  # Версионирование для резервного копирования
  versioning {
    enabled = true
  }

  # Жизненный цикл объектов
  lifecycle_rule {
    id      = "model-lifecycle"
    enabled = true

    # Переместить старые модели в холодное хранилище через 30 дней
    transition {
      days          = 30
      storage_class = "COLD"
    }

    # Удалить старые версии моделей через 90 дней
    noncurrent_version_expiration {
      days = 90
    }
  }

  # Шифрование на стороне сервера
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = var.tags
}

# Bucket для данных и логов
resource "yandex_storage_bucket" "data" {
  bucket = "${var.name_prefix}-data"
  
  # Жизненный цикл для логов
  lifecycle_rule {
    id      = "log-lifecycle"
    enabled = true

    expiration {
      days = 365  # Хранить логи 1 год
    }
  }

  tags = var.tags
}

# Bucket для конфигураций и скриптов
resource "yandex_storage_bucket" "configs" {
  bucket = "${var.name_prefix}-configs"
  
  tags = var.tags
}

# Статический ключ доступа для приложений
resource "yandex_iam_service_account_static_access_key" "bucket_access" {
  service_account_id = yandex_iam_service_account.storage_user.id
  description        = "Access key for ML models bucket"
}

# Service Account для доступа к storage
resource "yandex_iam_service_account" "storage_user" {
  name        = "${var.name_prefix}-storage-sa"
  description = "Service account for storage access"
}

# Политики доступа
resource "yandex_resourcemanager_folder_iam_member" "storage_admin" {
  folder_id = var.folder_id
  role      = "storage.admin"
  member    = "serviceAccount:${yandex_iam_service_account.storage_user.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "storage_viewer" {
  folder_id = var.folder_id
  role      = "storage.viewer"
  member    = "serviceAccount:${yandex_iam_service_account.storage_user.id}"
}