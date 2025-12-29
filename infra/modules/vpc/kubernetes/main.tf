resource "yandex_kubernetes_cluster" "cluster" {
  name        = "${var.name_prefix}-k8s-cluster"
  description = "Kubernetes cluster for Credit Scoring ML serving"
  network_id  = var.vpc_id
  
  master {
    version   = var.k8s_version
    public_ip = true
    
    master_location {
      zone      = var.zone
      subnet_id = var.subnet_ids[0]
    }
    
    security_group_ids = [var.security_group_id]
  }
  
  service_account_id      = yandex_iam_service_account.k8s.id
  node_service_account_id = yandex_iam_service_account.k8s.id
  
  kms_provider {
    key_id = yandex_kms_symmetric_key.k8s.id
  }
  
  labels = var.tags
}

resource "yandex_kubernetes_node_group" "nodes" {
  cluster_id  = yandex_kubernetes_cluster.cluster.id
  name        = "${var.name_prefix}-node-group"
  description = "Node group for ML workloads"
  
  instance_template {
    platform_id = "standard-v2"
    
    resources {
      memory = var.node_memory * 1024  # Convert GB to MB
      cores  = var.node_cpu
    }
    
    boot_disk {
      type = "network-ssd"
      size = var.node_disk_size
    }
    
    scheduling_policy {
      preemptible = var.environment != "prod"
    }
    
    network_interface {
      subnet_ids = var.subnet_ids
      nat        = true
    }
    
    container_runtime {
      type = "containerd"
    }
  }
  
  scale_policy {
    fixed_scale {
      size = var.node_count
    }
  }
  
  allocation_policy {
    location {
      zone = var.zone
    }
  }
  
  maintenance_policy {
    auto_upgrade = true
    auto_repair  = true
    
    maintenance_window {
      start_time = "03:00"
      duration   = "3h"
    }
  }
  
  labels = var.tags
}

resource "yandex_iam_service_account" "k8s" {
  name        = "${var.name_prefix}-k8s-sa"
  description = "Service account for Kubernetes"
  folder_id   = var.folder_id
}

resource "yandex_kms_symmetric_key" "k8s" {
  name        = "${var.name_prefix}-k8s-key"
  description = "KMS key for Kubernetes secrets"
  folder_id   = var.folder_id
}

resource "yandex_resourcemanager_folder_iam_member" "k8s_roles" {
  for_each = toset([
    "container-registry.images.puller",
    "monitoring.editor",
    "logging.writer",
    "k8s.clusters.agent",
    "vpc.publicAdmin",
    "load-balancer.admin",
    "certificate-manager.certificates.downloader"
  ])
  
  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.k8s.id}"
}