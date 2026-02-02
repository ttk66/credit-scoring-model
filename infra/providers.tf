provider "yandex" {
  zone      = var.yc_zone
  folder_id = var.yc_folder_id
  token     = var.yc_token
}

data "yandex_client_config" "this" {}

provider "kubernetes" {
  host                   = module.kubernetes.cluster_external_v4_endpoint
  cluster_ca_certificate = module.kubernetes.cluster_ca_certificate
  token                  = data.yandex_client_config.this.iam_token
}

provider "helm" {
  kubernetes {
    host                   = module.kubernetes.cluster_external_v4_endpoint
    cluster_ca_certificate = module.kubernetes.cluster_ca_certificate
    token                  = data.yandex_client_config.this.iam_token
  }
}
