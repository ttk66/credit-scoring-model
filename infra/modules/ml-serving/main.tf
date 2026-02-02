# Namespace для ML приложений
resource "kubernetes_namespace_v1" "ml_serving" {
  metadata {
    name = "ml-serving"
    labels = {
      name = "ml-serving"
    }
  }
}

# ConfigMap с конфигурацией модели
resource "kubernetes_config_map_v1" "model_config" {
  metadata {
    name      = "model-config"
    namespace = kubernetes_namespace_v1.ml_serving.metadata[0].name
  }

  data = {
    "model-config.yaml" = <<-EOT
      model:
        name: "credit-scoring"
        version: "1.0.0"
        framework: "onnx"
        input_shape: [1, 32]
        output_shape: [1, 2]
      
      inference:
        batch_size: 32
        timeout_ms: 5000
        max_concurrent_requests: 100
      
      monitoring:
        enabled: true
        metrics_port: 9090
        endpoint: "/metrics"
      
      autoscaling:
        min_replicas: 2
        max_replicas: 10
        target_cpu_utilization: 70
        target_memory_utilization: 80
    EOT
  }
}

# Secret с ключами доступа к storage
resource "kubernetes_secret_v1" "storage_credentials" {
  metadata {
    name      = "storage-credentials"
    namespace = kubernetes_namespace_v1.ml_serving.metadata[0].name
  }

  data = {
    "access-key" = var.storage_access_key
    "secret-key" = var.storage_secret_key
  }

  type = "Opaque"
}

# Deployment для ML serving
resource "kubernetes_deployment_v1" "ml_api" {
  metadata {
    name      = "credit-scoring-api"
    namespace = kubernetes_namespace_v1.ml_serving.metadata[0].name
    labels = {
      app = "credit-scoring-api"
    }
  }

  spec {
    replicas = 2

    selector {
      match_labels = {
        app = "credit-scoring-api"
      }
    }

    template {
      metadata {
        labels = {
          app = "credit-scoring-api"
        }
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "9090"
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        service_account_name = kubernetes_service_account_v1.ml_serving.metadata[0].name
        
        container {
          name  = "api"
          image = "cr.yandex/${var.container_registry_id}/credit-scoring-api:latest"
          image_pull_policy = "Always"

          port {
            container_port = 8000
            name = "http"
          }

          port {
            container_port = 9090
            name = "metrics"
          }

          env {
            name  = "MODEL_PATH"
            value = "/models/nn_model.onnx"
          }

          env {
            name  = "BATCH_SIZE"
            value = "32"
          }

          env {
            name = "STORAGE_ACCESS_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret_v1.storage_credentials.metadata[0].name
                key  = "access-key"
              }
            }
          }

          env {
            name = "STORAGE_SECRET_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret_v1.storage_credentials.metadata[0].name
                key  = "secret-key"
              }
            }
          }

          volume_mount {
            name       = "models-volume"
            mount_path = "/models"
            read_only  = true
          }

          volume_mount {
            name       = "config-volume"
            mount_path = "/app/config"
            read_only  = true
          }

          resources {
            requests = {
              cpu    = "200m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "1000m"
              memory = "512Mi"
            }
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 30
            period_seconds        = 10
          }

          readiness_probe {
            http_get {
              path = "/ready"
              port = 8000
            }
            initial_delay_seconds = 5
            period_seconds        = 5
          }
        }

        init_container {
          name  = "model-downloader"
          image = "cr.yandex/${var.container_registry_id}/model-downloader:latest"
          image_pull_policy = "Always"

          env {
            name  = "MODEL_BUCKET"
            value = var.model_bucket_name
          }

          env {
            name  = "MODEL_KEY"
            value = "nn_model.onnx"
          }

          env {
            name  = "MODEL_PATH"
            value = "/models/nn_model.onnx"
          }

          env {
            name = "STORAGE_ACCESS_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret_v1.storage_credentials.metadata[0].name
                key  = "access-key"
              }
            }
          }

          env {
            name = "STORAGE_SECRET_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret_v1.storage_credentials.metadata[0].name
                key  = "secret-key"
              }
            }
          }

          volume_mount {
            name       = "models-volume"
            mount_path = "/models"
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "128Mi"
            }
            limits = {
              cpu    = "200m"
              memory = "256Mi"
            }
          }
        }

        volume {
          name = "models-volume"
          empty_dir {}
        }

        volume {
          name = "config-volume"
          config_map {
            name = kubernetes_config_map_v1.model_config.metadata[0].name
          }
        }
      }
    }
  }
}

# Service для доступа к API
resource "kubernetes_service_v1" "ml_api" {
  metadata {
    name      = "credit-scoring-api"
    namespace = kubernetes_namespace_v1.ml_serving.metadata[0].name
  }

  spec {
    selector = {
      app = "credit-scoring-api"
    }

    port {
      name        = "http"
      port        = 80
      target_port = 8000
    }

    port {
      name        = "metrics"
      port        = 9090
      target_port = 9090
    }

    type = "ClusterIP"
  }
}

# Ingress для внешнего доступа
resource "kubernetes_ingress_v1" "ml_api" {
  metadata {
    name      = "credit-scoring-ingress"
    namespace = kubernetes_namespace_v1.ml_serving.metadata[0].name
    annotations = {
      "kubernetes.io/ingress.class" = "nginx"
      "nginx.ingress.kubernetes.io/ssl-redirect" = "true"
      "nginx.ingress.kubernetes.io/proxy-body-size" = "10m"
    }
  }

  spec {
    rule {
      host = var.api_domain
      http {
        path {
          path = "/"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service_v1.ml_api.metadata[0].name
              port {
                number = 80
              }
            }
          }
        }
      }
    }

    tls {
      hosts = [var.api_domain]
      secret_name = "tls-secret"
    }
  }
}

# Horizontal Pod Autoscaler
resource "kubernetes_horizontal_pod_autoscaler_v2" "ml_api" {
  metadata {
    name      = "credit-scoring-api-hpa"
    namespace = kubernetes_namespace_v1.ml_serving.metadata[0].name
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment_v1.ml_api.metadata[0].name
    }

    min_replicas = 2
    max_replicas = 10

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
      }
    }
  }
}

# Service Account для ML serving
resource "kubernetes_service_account_v1" "ml_serving" {
  metadata {
    name      = "ml-serving-sa"
    namespace = kubernetes_namespace_v1.ml_serving.metadata[0].name
  }

  automount_service_account_token = true
}
