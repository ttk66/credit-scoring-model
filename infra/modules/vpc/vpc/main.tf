resource "yandex_vpc_network" "main" {
  name        = "${var.name_prefix}-network"
  description = "VPC for Credit Scoring project"
  labels      = var.tags
}

resource "yandex_vpc_subnet" "subnets" {
  count          = length(var.zones)
  name           = "${var.name_prefix}-subnet-${count.index + 1}"
  zone           = var.zones[count.index]
  network_id     = yandex_vpc_network.main.id
  v4_cidr_blocks = [cidrsubnet(var.vpc_cidr, 8, count.index * 32)]
  
  labels = var.tags
}

resource "yandex_vpc_security_group" "k8s" {
  name        = "${var.name_prefix}-k8s-security-group"
  description = "Security group for Kubernetes cluster"
  network_id  = yandex_vpc_network.main.id
  
  labels = var.tags

  ingress {
    protocol       = "TCP"
    description    = "Kubernetes API"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 6443
  }

  ingress {
    protocol       = "TCP"
    description    = "HTTPS"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 443
  }

  ingress {
    protocol       = "TCP"
    description    = "HTTP"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 80
  }

  ingress {
    protocol       = "TCP"
    description    = "SSH"
    v4_cidr_blocks = ["0.0.0.0/0"]
    port           = 22
  }

  ingress {
    protocol          = "TCP"
    description       = "NodePort services"
    v4_cidr_blocks    = ["0.0.0.0/0"]
    from_port         = 30000
    to_port           = 32767
  }

  egress {
    protocol       = "ANY"
    description    = "Outgoing traffic"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}