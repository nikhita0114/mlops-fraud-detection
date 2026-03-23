variable "project_id" {
  description = "GCP project ID — find this in the GCP console"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"

  validation {
    condition = contains([
      "us-central1", "us-east1", "us-west1",
      "europe-west1", "europe-west2",
      "asia-east1", "asia-southeast1"
    ], var.region)
    error_message = "Region must be a valid GCP region."
  }
}

variable "environment" {
  description = "Deployment environment — used to prefix resource names"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "machine_type" {
  description = "GCE machine type for cluster nodes"
  type        = string
  default     = "e2-standard-2"
}

variable "min_node_count" {
  description = "Minimum number of nodes in the node pool (autoscaling lower bound)"
  type        = number
  default     = 1

  validation {
    condition     = var.min_node_count >= 1
    error_message = "min_node_count must be at least 1."
  }
}

variable "max_node_count" {
  description = "Maximum number of nodes in the node pool (autoscaling upper bound)"
  type        = number
  default     = 3

  validation {
    condition     = var.max_node_count >= 1
    error_message = "max_node_count must be at least 1."
  }
}

variable "use_spot_instances" {
  description = "Use spot/preemptible instances for ~60-90% cost savings (can be terminated anytime)"
  type        = bool
  default     = true
}

variable "model_bucket_location" {
  description = "GCS bucket location for model artifacts"
  type        = string
  default     = "US"
}

variable "artifact_registry_location" {
  description = "Location for Artifact Registry (Docker images)"
  type        = string
  default     = "us-central1"
}
