# =============================================================================
# storage.tf — GCS buckets
# =============================================================================
#
# Two buckets:
#   1. Model artifacts — stores fraud_model.pkl, versioned by git SHA
#   2. Terraform state — stores terraform.tfstate (created manually first)
#
# Why store models in GCS?
# The model .pkl file is gitignored (binary, too large for git).
# In production, CI trains the model and uploads it to GCS tagged with
# the git SHA. The K8s deployment downloads the correct version at startup.
# This gives you model versioning and rollback without rebuilding images.
# =============================================================================

# ---------------------------------------------------------------------------
# Model artifacts bucket
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "model_artifacts" {
  name          = "${var.project_id}-${var.environment}-fraud-models"
  location      = var.model_bucket_location
  force_destroy = var.environment != "prod"   # protect prod bucket from accidental deletion

  # Versioning — keeps previous versions of model files
  # Allows rollback to a previous model without retraining
  versioning {
    enabled = true
  }

  # Lifecycle rules — automatically manage storage costs
  lifecycle_rule {
    # Move old model versions to cheaper storage after 30 days
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age                   = 30           # days
      with_state            = "ARCHIVED"   # only apply to non-current versions
    }
  }

  lifecycle_rule {
    # Delete very old model versions after 365 days
    action {
      type = "Delete"
    }
    condition {
      age        = 365
      with_state = "ARCHIVED"
    }
  }

  # Uniform bucket-level access — simpler IAM, no per-object ACLs
  uniform_bucket_level_access = true

  labels = {
    environment = var.environment
    app         = "fraud-detection"
    managed-by  = "terraform"
  }
}

# IAM — allow GKE pods (via Workload Identity) to read models
resource "google_storage_bucket_iam_member" "model_reader" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.project_id}.svc.id.goog[mlops-prod/fraud-detection-sa]"
}

# IAM — allow CI/CD (GitHub Actions) to upload new models
# In practice, bind this to a dedicated CI service account
resource "google_storage_bucket_iam_member" "model_writer" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:ci-cd@${var.project_id}.iam.gserviceaccount.com"
}

# ---------------------------------------------------------------------------
# Terraform state bucket
# ---------------------------------------------------------------------------
# NOTE: This bucket must be created MANUALLY before running terraform init,
# because Terraform can't use this bucket to store state until it exists.
# Create it once with: gsutil mb gs://fraud-api-tfstate
#
# This resource documents it in code for visibility — Terraform will
# import it rather than create it if it already exists.

resource "google_storage_bucket" "terraform_state" {
  name          = "fraud-api-tfstate"
  location      = "US"
  force_destroy = false   # NEVER auto-delete the state bucket

  versioning {
    enabled = true   # keeps state history — essential for recovery
  }

  # State files can contain sensitive values — enforce encryption and access control
  uniform_bucket_level_access = true

  labels = {
    purpose    = "terraform-state"
    managed-by = "terraform"
  }
}
