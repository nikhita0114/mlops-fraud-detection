# MLOps Fraud Detection System

End-to-end MLOps platform for real-time fraud detection — built to demonstrate production-grade DevOps practices across the full stack: containerisation, CI/CD, Kubernetes, observability, drift detection, and infrastructure as code.

---

## What This Project Demonstrates

| Area | Implementation |
|---|---|
| Containerisation | Non-root Docker image, `.dockerignore`, multi-stage-ready, `HEALTHCHECK` |
| CI/CD | 5-job GitHub Actions pipeline — lint, test, audit, terraform, build+scan+push |
| Security | Trivy CVE scanning, pip-audit dependency audit, pinned deps, secrets in `.env` |
| Kubernetes | Namespace, ClusterIP, Ingress, ConfigMap, Secret, HPA, PDB, `securityContext` |
| Observability | Prometheus metrics, Grafana dashboards provisioned as code, alerting rules |
| Structured logging | JSON logs via `python-json-logger` — parseable by Loki, Datadog, CloudWatch |
| Drift detection | Evidently-powered `/drift-report` endpoint with rolling window + Prometheus gauge |
| IaC | Terraform for GKE cluster, Artifact Registry, GCS model storage, remote state |
| Testing | 23 pytest tests covering API, risk logic, input validation, drift detection |

---

## Architecture

```
Developer → git push
           ↓
GitHub Actions (lint → test → pip-audit → terraform validate → docker build+push)
           ↓
GHCR (image:sha + image:main)
           ↓
GKE Cluster (mlops-prod namespace)
├── Ingress (nginx + TLS)
├── Service (ClusterIP)
├── Deployment (2 replicas, HPA 2–5, PDB minAvailable:1)
│   └── FastAPI pods (/predict, /health, /metrics, /drift-report)
└── Observability
    ├── Prometheus (scrapes /metrics every 15s)
    ├── Grafana (provisioned as code — dashboards + datasources)
    └── Evidently (drift detection on rolling 100-transaction window)
```

Infrastructure provisioned by Terraform (GKE, Artifact Registry, GCS).

---

## CI/CD Pipeline

The pipeline has 5 sequential jobs — every step must pass before the next runs:

```
Lint (ruff) → Test (pytest 23 tests) → Dependency audit (pip-audit)
                                              ↓
                                  Terraform validate (fmt + init + validate)
                                              ↓
                                  Build + Trivy CVE scan + Push to GHCR
```

Pull requests are gated on lint + test + audit — broken code cannot merge to `main`.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check — used by K8s probes |
| `/predict` | POST | Fraud prediction with risk explanation |
| `/drift-report` | GET | Statistical drift analysis vs training baseline |
| `/metrics` | GET | Prometheus metrics endpoint |

### Example Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 95000,
    "time_of_day": 3,
    "distance_from_home_km": 400,
    "transactions_today": 10
  }'
```

```json
{
  "prediction": "fraud",
  "fraud_probability": "94.2%",
  "risk_factors": [
    "⚠️ Transaction at unusual hour",
    "⚠️ High transaction amount",
    "⚠️ Far from home location",
    "⚠️ Many transactions today"
  ]
}
```

---

## Quick Start (Local)

```bash
# 1. Clone and set up
git clone https://github.com/nikhita0114/mlops-fraud-detection.git
cd mlops-fraud-detection

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 3. Copy secrets template
cp .env.example .env  # edit with your local values

# 4. Run tests
pytest tests/ -v

# 5. Start full stack (API + Prometheus + Grafana)
docker-compose up

# 6. Open Grafana dashboard (auto-provisioned)
open http://localhost:3000  # login: admin / your .env password

# 7. Run drift simulation — watch the Grafana fraud rate panel spike
python drift_simulation.py

# 8. Check drift report
curl http://localhost:8000/drift-report | python3 -m json.tool
```

---

## Monitoring

Grafana dashboard is provisioned automatically on startup — no manual setup required.

### Prometheus Alert Rules (`monitoring/alerts.yml`)

| Alert | Condition | Severity |
|---|---|---|
| `FraudAPIDown` | No predictions for 1 minute | critical |
| `HighFraudRate` | Fraud rate > 50% for 2 minutes | warning |
| `HighPredictionLatency` | P95 latency > 500ms for 2 minutes | warning |
| `NoTraffic` | Zero predictions for 5 minutes | warning |

---

## Drift Detection

The `/drift-report` endpoint compares recent production transactions against the training distribution using Evidently:

```bash
curl http://localhost:8000/drift-report
```

```json
{
  "drift_detected": true,
  "drifted_features": ["amount", "distance_from_home_km"],
  "share_drifted": 0.5,
  "per_feature": {
    "amount": { "drift_detected": true, "drift_score": 0.001, "stattest": "ks" },
    "time_of_day": { "drift_detected": false, "drift_score": 0.312, "stattest": "ks" }
  },
  "current_window_size": 60
}
```

Drift signals are also exposed as Prometheus gauges (`drift_detected{feature="amount"}`) so Grafana can graph them over time.

---

## Infrastructure (Terraform)

The `terraform/` directory provisions the full GCP infrastructure:

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars  # fill in your project ID
terraform init
terraform plan
terraform apply
```

Resources created:
- GKE cluster with autoscaling node pool (spot instances for cost savings)
- Google Artifact Registry for Docker images
- GCS bucket for model artifacts (versioned, lifecycle policies)
- Remote state stored in GCS

---

## Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI app — predict, drift-report, metrics endpoints
│   └── drift.py         # Evidently drift detection logic
├── model/
│   └── train.py         # Model training script
├── tests/
│   ├── test_api.py      # 16 API tests
│   └── test_drift.py    # 7 drift detection tests
├── k8s/                 # Kubernetes manifests
│   ├── namespace.yml
│   ├── deployment.yml   # securityContext, HPA, PDB
│   ├── service.yml      # ClusterIP
│   ├── ingress.yml      # nginx + TLS
│   ├── configmap.yml
│   ├── secret.yml
│   ├── hpa.yml
│   └── pdb.yml
├── monitoring/
│   ├── prometheus.yml   # scrape config + alert rules reference
│   ├── alerts.yml       # 4 Prometheus alert rules
│   └── grafana/
│       ├── datasources/ # auto-provisioned Prometheus datasource
│       └── dashboards/  # auto-provisioned fraud API dashboard
├── terraform/           # GKE, Artifact Registry, GCS
├── Dockerfile           # non-root user, HEALTHCHECK, exec CMD
├── docker-compose.yml   # api + prometheus + grafana, named volumes
└── .github/workflows/
    └── deploy.yml       # 5-job CI/CD pipeline
```

---

## Tech Stack

FastAPI · scikit-learn · Evidently · Prometheus · Grafana · Docker · Kubernetes · GitHub Actions · Trivy · Terraform · GCP (GKE, Artifact Registry, GCS) · Python 3.11
