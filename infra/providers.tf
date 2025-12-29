provider "yandex" {
  zone      = var.yc_zone
  folder_id = var.yc_folder_id
  token     = var.yc_token
}

provider "kubernetes" {
  host                   = module.kubernetes.cluster_external_v4_endpoint
  cluster_ca_certificate = module.kubernetes.cluster_ca_certificate
  token                  = var.service_account_token
}

provider "helm" {
  kubernetes {
    host                   = module.kubernetes.cluster_external_v4_endpoint
    cluster_ca_certificate = module.kubernetes.cluster_ca_certificate
    token                  = var.service_account_token
  }
}