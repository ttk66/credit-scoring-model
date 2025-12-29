output "cluster_id" {
  description = "Kubernetes cluster ID"
  value       = yandex_kubernetes_cluster.cluster.id
}

output "cluster_external_v4_endpoint" {
  description = "External endpoint for Kubernetes API"
  value       = yandex_kubernetes_cluster.cluster.master[0].external_v4_endpoint
}

output "cluster_ca_certificate" {
  description = "CA certificate for Kubernetes cluster"
  value       = yandex_kubernetes_cluster.cluster.master[0].cluster_ca_certificate
  sensitive   = true
}

output "node_group_id" {
  description = "Node group ID"
  value       = yandex_kubernetes_node_group.nodes.id
}