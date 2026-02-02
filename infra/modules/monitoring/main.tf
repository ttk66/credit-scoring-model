# Monitoring service account
resource "yandex_iam_service_account" "monitoring" {
  name        = "${var.name_prefix}-monitoring-sa"
  description = "Service account for monitoring"
}

#   monitoring service account
resource "yandex_resourcemanager_folder_iam_member" "monitoring_roles" {
  for_each = toset([
    "monitoring.editor",
    "monitoring.viewer",
    "logging.writer",
    "audit-trail.viewer"
  ])
  
  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.monitoring.id}"
}

#   Monitoring
resource "yandex_monitoring_dashboard" "credit_scoring" {
  name        = "${var.name_prefix}-dashboard"
  description = "Credit Scoring ML Dashboard"

  widget {
    position {
      h = 6
      w = 6
      x = 0
      y = 0
    }
    
    title = "ML Model Performance"
    
    chart {
      name = "model_inference_latency"
      title = "Inference Latency (ms)"
      
      queries {
        target = <<-EOT
          avg(rate(
            yc_mdb_clickhouse_UserMetrics_Query{service="credit-scoring",metric="inference_latency"}
          [5m]))
        EOT
      }
      
      visualization_settings {
        type = "VISUALIZATION_TYPE_LINE"
        color_scheme_settings {
          automatic {}
        }
      }
    }
  }

  widget {
    position {
      h = 6
      w = 6
      x = 6
      y = 0
    }
    
    title = "API Requests"
    
    chart {
      name = "api_requests"
      title = "Requests per Second"
      
      queries {
        target = <<-EOT
          sum(rate(
            yc_mdb_clickhouse_UserMetrics_Query{service="credit-scoring",metric="requests_total"}
          [5m]))
        EOT
      }
      
      visualization_settings {
        type = "VISUALIZATION_TYPE_LINE"
        color_scheme_settings {
          automatic {}
        }
      }
    }
  }

  widget {
    position {
      h = 6
      w = 6
      x = 0
      y = 6
    }
    
    title = "Error Rate"
    
    chart {
      name = "error_rate"
      title = "Error Rate (%)"
      
      queries {
        target = <<-EOT
          100 * sum(rate(
            yc_mdb_clickhouse_UserMetrics_Query{service="credit-scoring",metric="errors_total"}
          [5m])) / sum(rate(
            yc_mdb_clickhouse_UserMetrics_Query{service="credit-scoring",metric="requests_total"}
          [5m]))
        EOT
      }
      
      visualization_settings {
        type = "VISUALIZATION_TYPE_LINE"
        color_scheme_settings {
          automatic {}
        }
      }
    }
  }

  widget {
    position {
      h = 6
      w = 6
      x = 6
      y = 6
    }
    
    title = "CPU Usage"
    
    chart {
      name = "cpu_usage"
      title = "CPU Usage (%)"
      
      queries {
        target = <<-EOT
          100 - avg(rate(
            node_cpu_seconds_total{mode="idle", cluster="${var.k8s_cluster_id}"}
          [5m])) * 100
        EOT
      }
      
      visualization_settings {
        type = "VISUALIZATION_TYPE_LINE"
        color_scheme_settings {
          automatic {}
        }
      }
    }
  }

  widget {
    position {
      h = 6
      w = 12
      x = 0
      y = 12
    }
    
    title = "Memory Usage"
    
    chart {
      name = "memory_usage"
      title = "Memory Usage (GB)"
      
      queries {
        target = <<-EOT
          (node_memory_MemTotal_bytes{cluster="${var.k8s_cluster_id}"} 
          - node_memory_MemFree_bytes{cluster="${var.k8s_cluster_id}"} 
          - node_memory_Buffers_bytes{cluster="${var.k8s_cluster_id}"} 
          - node_memory_Cached_bytes{cluster="${var.k8s_cluster_id}"}) / 1024 / 1024 / 1024
        EOT
      }
      
      visualization_settings {
        type = "VISUALIZATION_TYPE_LINE"
        color_scheme_settings {
          automatic {}
        }
      }
    }
  }

  tags = var.tags
}

# 
resource "yandex_monitoring_alert" "high_error_rate" {
  name        = "${var.name_prefix}-high-error-rate"
  description = "High error rate in credit scoring API"
  
  labels = var.tags
  
  alert_rule {
    rule_type = "STATUS_ALERT"
    
    triggers {
      type = "METRIC"
      
      metric_trigger {
        metric {
          type  = "YANDEX_MONITORING"
          labels = {
            service = "credit-scoring"
            metric  = "error_rate"
          }
        }
        
        condition {
          evaluation_window = "EVALUATION_WINDOW_LAST_5M"
          condition_type    = "CONDITION_TYPE_GREATER"
          threshold         = 5.0  # 5% error rate
        }
      }
    }
    
    notification_channels = [yandex_monitoring_notification_channel.email.id]
    
    alert_strategy {
      auto_close = "AUTO_CLOSE_AFTER_1H"
    }
  }
}

resource "yandex_monitoring_alert" "high_latency" {
  name        = "${var.name_prefix}-high-latency"
  description = "High inference latency"
  
  labels = var.tags
  
  alert_rule {
    rule_type = "STATUS_ALERT"
    
    triggers {
      type = "METRIC"
      
      metric_trigger {
        metric {
          type  = "YANDEX_MONITORING"
          labels = {
            service = "credit-scoring"
            metric  = "inference_latency"
          }
        }
        
        condition {
          evaluation_window = "EVALUATION_WINDOW_LAST_5M"
          condition_type    = "CONDITION_TYPE_GREATER"
          threshold         = 1000.0  # 1000ms
        }
      }
    }
    
    notification_channels = [yandex_monitoring_notification_channel.email.id]
    
    alert_strategy {
      auto_close = "AUTO_CLOSE_AFTER_1H"
    }
  }
}

resource "yandex_monitoring_alert" "low_success_rate" {
  name        = "${var.name_prefix}-low-success-rate"
  description = "Low prediction success rate"
  
  labels = var.tags
  
  alert_rule {
    rule_type = "STATUS_ALERT"
    
    triggers {
      type = "METRIC"
      
      metric_trigger {
        metric {
          type  = "YANDEX_MONITORING"
          labels = {
            service = "credit-scoring"
            metric  = "success_rate"
          }
        }
        
        condition {
          evaluation_window = "EVALUATION_WINDOW_LAST_5M"
          condition_type    = "CONDITION_TYPE_LESS"
          threshold         = 95.0  # 95% success rate
        }
      }
    }
    
    notification_channels = [yandex_monitoring_notification_channel.email.id]
    
    alert_strategy {
      auto_close = "AUTO_CLOSE_AFTER_1H"
    }
  }
}

#  
resource "yandex_monitoring_notification_channel" "email" {
  name        = "${var.name_prefix}-email-notifications"
  description = "Email notifications for ML team"
  
  labels = var.tags
  
  email_settings {
    recipients = var.notification_emails
  }
}

resource "yandex_monitoring_notification_channel" "telegram" {
  count       = var.telegram_bot_token != "" ? 1 : 0
  name        = "${var.name_prefix}-telegram-notifications"
  description = "Telegram notifications for on-call"
  
  labels = var.tags
  
  telegram_settings {
    bot_token = var.telegram_bot_token
    chat_id   = var.telegram_chat_id
  }
}

#  (Cloud Logging)
resource "yandex_logging_group" "ml_logs" {
  name        = "${var.name_prefix}-ml-logs"
  description = "Log group for credit scoring ML service"
  
  retention_period = "2592000000000"  # 30 days in nanoseconds
  
  labels = var.tags
}

resource "yandex_logging_group" "audit_logs" {
  name        = "${var.name_prefix}-audit-logs"
  description = "Audit logs for compliance"
  
  retention_period = "7776000000000"  # 90 days in nanoseconds
  
  labels = var.tags
}