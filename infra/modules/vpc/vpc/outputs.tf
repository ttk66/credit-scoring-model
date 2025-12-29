output "vpc_id" {
  description = "ID of the created VPC"
  value       = yandex_vpc_network.main.id
}

output "subnet_ids" {
  description = "IDs of the created subnets"
  value       = yandex_vpc_subnet.subnets[*].id
}

output "security_group_id" {
  description = "ID of the Kubernetes security group"
  value       = yandex_vpc_security_group.k8s.id
}