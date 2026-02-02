module "vpc_network" {
  source = "./vpc"
  
  name_prefix = var.name_prefix
  vpc_cidr    = var.vpc_cidr
  tags        = var.tags
  zones       = var.zones
}
