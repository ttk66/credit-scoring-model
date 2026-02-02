locals {
  # Common tags (lowercase for Yandex labels)
  tags = {
    project     = var.project_name
    environment = var.environment
    managed-by  = "terraform"
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

# TODO: bucket для remote state требует специальных прав storage.admin
# Может быть создан отдельно через yc CLI или через Service Account с правами storage.admin
# resource "yandex_storage_bucket" "terraform_state" {
#   bucket     = "${local.name_prefix}-terraform-state"
#   access_key = yandex_iam_service_account_static_access_key.terraform_key.access_key
#   secret_key = yandex_iam_service_account_static_access_key.terraform_key.secret_key
#
#   versioning {
#     enabled = true
#   }
#
#   lifecycle {
#     prevent_destroy = true
#   }
# }


# Подключаем модули
module "vpc" {
  source = "./modules/vpc"
  
  name_prefix = local.name_prefix
  vpc_cidr    = var.vpc_cidr
  tags        = local.tags
  zones       = [var.yc_zone]
}

module "kubernetes" {
  source = "./modules/vpc/kubernetes"
  
  name_prefix          = local.name_prefix
  vpc_id               = module.vpc.vpc_id
  subnet_ids           = module.vpc.subnet_ids
  folder_id            = var.yc_folder_id
  security_group_id    = module.vpc.security_group_id
  k8s_version          = var.k8s_version
  node_count           = var.k8s_node_count
  node_disk_size       = var.k8s_node_disk_size
  node_cpu             = var.k8s_node_cpu
  node_memory          = var.k8s_node_memory
  service_account_name = yandex_iam_service_account.terraform.name
  zone                 = var.yc_zone
  environment          = var.environment
  tags                 = local.tags
  
  depends_on = [module.vpc]
}

# TODO: Storage buckets требуют специальных прав storage.admin
# Будут созданы отдельно через yc CLI или через Service Account с правами
# module "storage" {
#   source = "./modules/vpc/storage"
#   
#   name_prefix = local.name_prefix
#   folder_id   = var.yc_folder_id
#   environment = var.environment
#   tags        = local.tags
# }

# TODO: Мониторинг будет добавлен на следующем этапе
# module "monitoring" {
#   source = "./modules/monitoring"
#   
#   name_prefix    = local.name_prefix
#   k8s_cluster_id = module.kubernetes.cluster_id
#   folder_id      = var.yc_folder_id
#   tags           = local.tags
#   
#   depends_on = [module.kubernetes]
# }


module "ml_serving" {
  source = "./modules/ml-serving"
  
  name_prefix           = local.name_prefix
  container_registry_id = "crp9o02lqmtgc663hs6c"
  storage_access_key    = var.yc_access_key
  storage_secret_key    = var.yc_secret_key
  model_bucket_name     = "credit-scoring-dev-models"  # TODO: создать через yc CLI
  api_domain            = "api.${var.project_name}.example.com"
  tags                  = local.tags
  
  depends_on = [module.kubernetes]
}
