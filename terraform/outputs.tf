# =============================================================================
# outputs.tf — Values exported after terraform apply
# =============================================================================
#
# Outputs are printed after `terraform apply` completes.
# They can also be read by other Terraform modules or CI/CD pipelines:
#   terraform output -raw cluster_name
#   terraform output -json > infra.json
#
# Use these outputs to configure kubectl and CI/CD:
#   gcloud container clusters get-credentials $(terraform output -raw cluster_name) \
#     --region $(terraform output -raw region) \
#     --project $(terraform output -raw project_id)
# =============================================================================

output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "cluster_name" {
  description = "GKE cluster name — use with gcloud container clusters get-credentials"
  value       = google_container_cluster.fraud_api.name
}

output "cluster_endpoint" {
  description = "GKE cluster API server endpoint"
  value       = google_container_cluster.fraud_api.endpoint
  sensitive   = true # marked sensitive — won't print in plain text logs
}

output "registry_url" {
  description = "Artifact Registry URL — use as Docker image prefix"
  # Format: LOCATION-docker.pkg.dev/PROJECT/REPOSITORY
  # Example: us-central1-docker.pkg.dev/my-project/prod-fraud-api
  value = "${var.artifact_registry_location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.fraud_api.repository_id}"
}

output "model_bucket_name" {
  description = "GCS bucket name for model artifacts"
  value       = google_storage_bucket.model_artifacts.name
}

output "model_bucket_url" {
  description = "GCS bucket URL for model artifacts — use with gsutil"
  value       = "gs://${google_storage_bucket.model_artifacts.name}"
}

output "kubectl_config_command" {
  description = "Run this command to configure kubectl after apply"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.fraud_api.name} --region ${var.region} --project ${var.project_id}"
}
