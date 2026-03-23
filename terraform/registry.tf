resource "google_artifact_registry_repository" "fraud_api" {
  location      = var.artifact_registry_location
  repository_id = "${var.environment}-fraud-api"
  description   = "Docker images for the Fraud Detection API (${var.environment})"
  format        = "DOCKER"

  docker_config {
    immutable_tags = false
  }

  labels = {
    environment = var.environment
    app         = "fraud-detection"
    managed-by  = "terraform"
  }
}

resource "google_artifact_registry_repository_iam_member" "gke_reader" {
  location   = google_artifact_registry_repository.fraud_api.location
  repository = google_artifact_registry_repository.fraud_api.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${var.project_id}.svc.id.goog[mlops-prod/fraud-detection-sa]"
}