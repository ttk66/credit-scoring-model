output "dashboard_url" {
  description = "URL of the monitoring dashboard"
  value       = "https://monitoring.cloud.yandex.ru/dashboards/${yandex_monitoring_dashboard.credit_scoring.id}"
}

output "dashboard_id" {
  description = "ID of the monitoring dashboard"
  value       = yandex_monitoring_dashboard.credit_scoring.id
}

output "alert_ids" {
  description = "IDs of created alerts"
  value = {
    high_error_rate   = yandex_monitoring_alert.high_error_rate.id
    high_latency      = yandex_monitoring_alert.high_latency.id
    low_success_rate  = yandex_monitoring_alert.low_success_rate.id
  }
}

output "notification_channels" {
  description = "Notification channel IDs"
  value = {
    email    = yandex_monitoring_notification_channel.email.id
    telegram = try(yandex_monitoring_notification_channel.telegram[0].id, null)
  }
}

output "log_group_ids" {
  description = "IDs of log groups"
  value = {
    ml_logs    = yandex_logging_group.ml_logs.id
    audit_logs = yandex_logging_group.audit_logs.id
  }
}

output "monitoring_service_account_id" {
  description = "ID of the monitoring service account"
  value       = yandex_iam_service_account.monitoring.id
}