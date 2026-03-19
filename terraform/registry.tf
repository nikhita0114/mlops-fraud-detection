# =============================================================================
# registry.tf — Artifact Registry for Docker images
# =============================================================================
#
# Artifact Registry replaces the older Container Registry (gcr.io).
# It stores Docker images, supports vulnerability scanning, and integrates
# with GKE for pulling images without extra credentials.
#
# Why not GHCR (GitHub Container Registry)?
# In production you'd use the registry closest to your cluster to minimize
# image pull latency. GKE pulling from Artifact Registry in the same region
# is faster and free (no egress costs). GHCR is fine for open source projects
# but Artifact Registry is the production choice for GKE workloads.
# =============================================================================

resource "google_artifact_registry_repository" "fraud_api" {
  location      = var.artifact_registry_location
  repository_id = "${var.environment}-fraud-api"
  description   = "Docker images for the Fraud Detection API (${var.environment})"
  format        = "DOCKER"

  # Vulnerability scanning — automatically scan pushed images for CVEs
  # Works alongside our Trivy scan in CI — defence in depth
  docker_config {
    immutable_tags = false   # allow overwriting tags (set true for strict prod)
  }

  labels = {
    environment = var.environment
    app         = "fraud-detection"
    managed-by  = "terraform"
  }

  # Cleanup policy — automatically delete old images to control storage costs
  # Keep the last 10 tagged images and delete untagged images after 7 days
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-tagged-releases"
    action = "KEEP"
    condition {
      tag_state             = "TAGGED"
      newer_version_count   = 10   # keep last 10 tagged images
    }
  }

  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s"   # 7 days in seconds
    }
  }
}

# IAM binding — allow GKE nodes to pull images from this registry
# Uses Workload Identity (defined in main.tf) so no service account keys needed
resource "google_artifact_registry_repository_iam_member" "gke_reader" {
  location   = google_artifact_registry_repository.fraud_api.location
  repository = google_artifact_registry_repository.fraud_api.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${var.project_id}.svc.id.goog[mlops-prod/fraud-detection-sa]"
}
