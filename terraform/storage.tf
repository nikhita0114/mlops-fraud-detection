resource "google_storage_bucket" "model_artifacts" {
  name          = "${var.project_id}-${var.environment}-fraud-models"
  location      = var.model_bucket_location
  force_destroy = var.environment != "prod"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age        = 30
      with_state = "ARCHIVED"
    }
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age        = 365
      with_state = "ARCHIVED"
    }
  }

  uniform_bucket_level_access = true

  labels = {
    environment = var.environment
    app         = "fraud-detection"
    managed-by  = "terraform"
  }
}

resource "google_storage_bucket_iam_member" "model_reader" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${var.project_id}.svc.id.goog[mlops-prod/fraud-detection-sa]"
}

resource "google_storage_bucket_iam_member" "model_writer" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:ci-cd@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_storage_bucket" "terraform_state" {
  name          = "fraud-api-tfstate"
  location      = "US"
  force_destroy = false

  versioning {
    enabled = true
  }

  uniform_bucket_level_access = true

  labels = {
    purpose    = "terraform-state"
    managed-by = "terraform"
  }
}