# =============================================================================
# main.tf — Provider configuration and GKE cluster
# =============================================================================
#
# This file defines:
#   1. Required providers and their versions (pinned for reproducibility)
#   2. Remote state backend (GCS bucket — shared across the team)
#   3. GKE cluster with a managed node pool
#
# Usage:
#   cp terraform.tfvars.example terraform.tfvars  # fill in your values
#   terraform init                                 # download providers
#   terraform plan                                 # preview changes
#   terraform apply                                # create infrastructure
#   terraform destroy                              # tear everything down
# =============================================================================

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"   # pinned — same reason we pin pip packages
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }

  # Remote state — stores terraform.tfstate in GCS instead of locally.
  # This means any team member can run terraform without conflicts.
  # The bucket is created manually once (chicken-and-egg problem with state).
  # See storage.tf for the bucket resource definition.
  backend "gcs" {
    bucket = "fraud-api-tfstate"   # must exist before terraform init
    prefix = "terraform/state"
  }
}

# Google Cloud provider — authenticates using Application Default Credentials
# Run: gcloud auth application-default login
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# GKE Cluster
# =============================================================================
#
# We create a "private cluster" — nodes have no public IPs, only the
# control plane endpoint is publicly accessible (with authorized networks).
# This follows GCP security best practices for production workloads.

resource "google_container_cluster" "fraud_api" {
  name     = "${var.environment}-fraud-api-cluster"
  location = var.region

  # We manage node pools separately (below) for more control
  # This creates an empty cluster with no default node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  # Network configuration
  network    = "default"
  subnetwork = "default"

  # Workload Identity — allows pods to authenticate to GCP services
  # (e.g. read model artifacts from GCS) without storing credentials in secrets
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Logging and monitoring — sends cluster logs to Cloud Logging / Monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  # Maintenance window — allow GKE to auto-upgrade during off-peak hours
  maintenance_policy {
    recurring_window {
      start_time = "2024-01-01T02:00:00Z"  # 2am UTC
      end_time   = "2024-01-01T06:00:00Z"  # 6am UTC
      recurrence = "FREQ=WEEKLY;BYDAY=SA"  # Saturday nights
    }
  }

  # Addons
  addons_config {
    # HTTP load balancing — required for Ingress to work
    http_load_balancing {
      disabled = false
    }
    # Horizontal pod autoscaling — required for HPA to work
    horizontal_pod_autoscaling {
      disabled = false
    }
  }
}

# =============================================================================
# Node Pool
# =============================================================================
#
# Separate node pool gives us:
#   - Independent scaling from the cluster control plane
#   - Ability to use preemptible/spot instances for cost savings
#   - Node auto-upgrade and auto-repair managed by GKE

resource "google_container_node_pool" "fraud_api_nodes" {
  name       = "${var.environment}-fraud-api-nodes"
  location   = var.region
  cluster    = google_container_cluster.fraud_api.name

  # Auto-scaling: scale between min and max based on pod demand
  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }

  # Auto-upgrade and auto-repair
  management {
    auto_repair  = true   # automatically replace unhealthy nodes
    auto_upgrade = true   # automatically upgrade node Kubernetes version
  }

  node_config {
    machine_type = var.machine_type  # e.g. "e2-standard-2"

    # Use spot instances to reduce cost by ~60-90%
    # Not suitable for stateful workloads — fine for stateless API pods
    spot = var.use_spot_instances

    # OAuth scopes — what GCP APIs the nodes can access
    oauth_scopes = [
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only",  # read model from GCS
    ]

    # Node labels — can be used for pod scheduling (nodeSelector)
    labels = {
      environment = var.environment
      app         = "fraud-detection"
    }

    # Node metadata
    metadata = {
      disable-legacy-endpoints = "true"   # security best practice
    }

    # Shielded instance config — protects against rootkit/bootkits
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}
