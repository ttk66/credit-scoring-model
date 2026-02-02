# Bucket   ML 
resource "yandex_storage_bucket" "models" {
  bucket = "${var.name_prefix}-models"
  
  # ACL  
  grant {
    type        = "Group"
    uri         = "http://acs.amazonaws.com/groups/global/AuthenticatedUsers"
    permissions = ["READ"]
  }

  #    
  versioning {
    enabled = true
  }

  #   
  lifecycle_rule {
    id      = "model-lifecycle"
    enabled = true

    #        30 
    transition {
      days          = 30
      storage_class = "COLD"
    }

    #      90 
    noncurrent_version_expiration {
      days = 90
    }
  }

  tags = var.tags
}

# Bucket    
resource "yandex_storage_bucket" "data" {
  bucket = "${var.name_prefix}-data"
  
  #    
  lifecycle_rule {
    id      = "log-lifecycle"
    enabled = true

    expiration {
      days = 365  #   1 
    }
  }

  tags = var.tags
}

# Bucket    
resource "yandex_storage_bucket" "configs" {
  bucket = "${var.name_prefix}-configs"
  
  tags = var.tags
}

#     
resource "yandex_iam_service_account_static_access_key" "bucket_access" {
  service_account_id = yandex_iam_service_account.storage_user.id
  description        = "Access key for ML models bucket"
}

# Service Account    storage
resource "yandex_iam_service_account" "storage_user" {
  name        = "${var.name_prefix}-storage-sa"
  description = "Service account for storage access"
}

#  
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