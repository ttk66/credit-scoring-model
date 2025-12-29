locals {
  # Common tags
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
    CreatedAt   = timestamp()
  }
  
  # Resource naming convention
  name_prefix = "${var.project_name}-${var.environment}"
}

# Создаем Service Account для Terraform
resource "yandex_iam_service_account" "terraform" {
  name        = "${local.name_prefix}-terraform-sa"
  description = "Service account for Terraform"
  folder_id   = var.yc_folder_id
}

# Назначаем роли Service Account
resource "yandex_resourcemanager_folder_iam_member" "editor" {
  folder_id = var.yc_folder_id
  role      = "editor"
  member    = "serviceAccount:${yandex_iam_service_account.terraform.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "vpc_admin" {
  folder_id = var.yc_folder_id
  role      = "vpc.admin"
  member    = "serviceAccount:${yandex_iam_service_account.terraform.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "k8s_admin" {
  folder_id = var.yc_folder_id
  role      = "k8s.admin"
  member    = "serviceAccount:${yandex_iam_service_account.terraform.id}"
}

# Создаем статический ключ для Service Account
resource "yandex_iam_service_account_static_access_key" "terraform_key" {
  service_account_id = yandex_iam_service_account.terraform.id
  description        = "Static access key for Terraform"
}

# Создаем bucket для remote state (должен быть создан заранее или через отдельный скрипт)
resource "yandex_storage_bucket" "terraform_state" {
  bucket     = "${local.name_prefix}-terraform-state"
  access_key = yandex_iam_service_account_static_access_key.terraform_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.terraform_key.secret_key

  versioning {
    enabled = true
  }

  lifecycle {
    prevent_destroy = true
  }
}

# Подключаем модули
module "vpc" {
  source = "./modules/vpc"
  
  name_prefix = local.name_prefix
  vpc_cidr    = var.vpc_cidr
  tags        = local.tags
}

module "kubernetes" {
  source = "./modules/kubernetes"
  
  name_prefix          = local.name_prefix
  vpc_id               = module.vpc.vpc_id
  subnet_ids           = module.vpc.subnet_ids
  k8s_version          = var.k8s_version
  node_count           = var.k8s_node_count
  node_disk_size       = var.k8s_node_disk_size
  node_cpu             = var.k8s_node_cpu
  node_memory          = var.k8s_node_memory
  service_account_name = yandex_iam_service_account.terraform.name
  tags                 = local.tags
  
  depends_on = [module.vpc]
}

module "storage" {
  source = "./modules/storage"
  
  name_prefix = local.name_prefix
  tags        = local.tags
}

module "monitoring" {
  source = "./modules/monitoring"
  
  name_prefix    = local.name_prefix
  k8s_cluster_id = module.kubernetes.cluster_id
  tags           = local.tags
  
  depends_on = [module.kubernetes]
}

module "ml_serving" {
  source = "./modules/ml-serving"
  
  name_prefix       = local.name_prefix
  k8s_cluster_endpoint = module.kubernetes.cluster_external_v4_endpoint
  k8s_cluster_ca    = module.kubernetes.cluster_ca_certificate
  model_bucket_name = module.storage.model_bucket_name
  tags              = local.tags
  
  depends_on = [module.kubernetes, module.storage]
}