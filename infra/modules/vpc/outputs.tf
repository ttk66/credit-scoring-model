output "vpc_id" {
  description = "ID of the created VPC"
  value       = module.vpc_network.vpc_id
}

output "subnet_ids" {
  description = "IDs of the created subnets"
  value       = module.vpc_network.subnet_ids
}

output "security_group_id" {
  description = "ID of the Kubernetes security group"
  value       = module.vpc_network.security_group_id
}
