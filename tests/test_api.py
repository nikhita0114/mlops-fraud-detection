"""
Tests for the Fraud Detection API.

We use FastAPI's built-in TestClient which runs the app in-process —
no need to start a real server. Tests run in milliseconds.

Structure:
  - test_health         → is the API alive?
  - test_predict_*      → does /predict return correct results?
  - test_risk_factors_* → does the risk explanation logic work?
  - test_input_*        → does the API handle bad input gracefully?
"""

from fastapi.testclient import TestClient

from app.main import app

# TestClient wraps the FastAPI app — simulates real HTTP requests
# without needing to spin up a server on a port
client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_returns_200():
    """
    The /health endpoint must always return 200.
    K8s liveness probe hits this — if it fails, the pod gets restarted.
    CI also hits this to confirm the app started correctly.
    """
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    """
    The body must contain {"status": "ok"}.
    Checking the body (not just status code) catches cases where
    the endpoint exists but is returning wrong data.
    """
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Predict — normal legitimate transaction
# ---------------------------------------------------------------------------

def test_predict_legitimate_transaction():
    """
    A low-risk transaction should return 200 and a valid prediction.
    We test that the response structure is correct and that fraud probability
    is LOWER for safe transactions than for high-risk ones.
    Note: the synthetic-data model may still flag some safe inputs —
    what matters is the API responds correctly, not the exact ML label.
    """
    response = client.post("/predict", json={
        "amount": 500.0,
        "time_of_day": 14,             # 2pm — normal hours
        "distance_from_home_km": 5.0,  # close to home
        "transactions_today": 1        # first transaction of the day
    })
    assert response.status_code == 200
    data = response.json()
    # Prediction must be one of the two valid labels
    assert data["prediction"] in ("legitimate", "fraud")
    # Probability must be a percentage string
    assert data["fraud_probability"].endswith("%")
    # Safe transaction must have lower fraud probability than an obvious attack
    safe_prob = float(data["fraud_probability"].replace("%", ""))

    attack_response = client.post("/predict", json={
        "amount": 95000.0,
        "time_of_day": 3,
        "distance_from_home_km": 400.0,
        "transactions_today": 10
    })
    attack_prob = float(attack_response.json()["fraud_probability"].replace("%", ""))
    assert safe_prob < attack_prob, (
        f"Safe transaction ({safe_prob}%) should have lower fraud probability "
        f"than an attack pattern ({attack_prob}%)"
    )


def test_predict_fraud_transaction():
    """
    A high-risk transaction (huge amount, 3am, 400km away, 10 txns today)
    should be predicted as fraud.
    """
    response = client.post("/predict", json={
        "amount": 95000.0,
        "time_of_day": 3,              # 3am — suspicious hour
        "distance_from_home_km": 400.0,  # very far from home
        "transactions_today": 10          # unusually many txns
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "fraud"


def test_predict_returns_probability():
    """
    The response must include a fraud_probability field.
    This is what Grafana graphs and what business teams use to set thresholds.
    """
    response = client.post("/predict", json={
        "amount": 1000.0,
        "time_of_day": 10,
        "distance_from_home_km": 10.0,
        "transactions_today": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    # Probability is returned as a string like "12.3%"
    assert data["fraud_probability"].endswith("%")


def test_predict_returns_risk_factors():
    """
    The response must include a risk_factors list.
    This is the human-readable explanation shown in the API response.
    """
    response = client.post("/predict", json={
        "amount": 1000.0,
        "time_of_day": 10,
        "distance_from_home_km": 10.0,
        "transactions_today": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "risk_factors" in data
    assert isinstance(data["risk_factors"], list)
    assert len(data["risk_factors"]) > 0


def test_predict_returns_transaction_summary():
    """
    The response must echo back the transaction details in a formatted summary.
    Useful for the API consumer to confirm what was sent.
    """
    response = client.post("/predict", json={
        "amount": 45000.0,
        "time_of_day": 2,
        "distance_from_home_km": 250.5,
        "transactions_today": 8
    })
    assert response.status_code == 200
    data = response.json()
    assert "transaction_summary" in data
    summary = data["transaction_summary"]
    assert "amount" in summary
    assert "time" in summary
    assert "distance_from_home" in summary
    assert "transactions_today" in summary


# ---------------------------------------------------------------------------
# Risk factor logic
# ---------------------------------------------------------------------------

def test_risk_factor_detected_for_night_transaction():
    """
    Transactions between 11pm and 6am should trigger the night risk factor.
    """
    response = client.post("/predict", json={
        "amount": 500.0,
        "time_of_day": 2,           # 2am
        "distance_from_home_km": 5.0,
        "transactions_today": 1
    })
    data = response.json()
    risk_text = " ".join(data["risk_factors"])
    assert "unusual hour" in risk_text.lower()


def test_risk_factor_detected_for_high_amount():
    """
    Transactions over ₹50,000 should trigger the high amount risk factor.
    """
    response = client.post("/predict", json={
        "amount": 75000.0,          # over the 50k threshold
        "time_of_day": 14,
        "distance_from_home_km": 5.0,
        "transactions_today": 1
    })
    data = response.json()
    risk_text = " ".join(data["risk_factors"])
    assert "high" in risk_text.lower()


def test_no_risk_factors_for_safe_transaction():
    """
    A completely normal transaction should return no risk factors
    — just the "no risk factors detected" message.
    """
    response = client.post("/predict", json={
        "amount": 200.0,
        "time_of_day": 12,           # noon
        "distance_from_home_km": 3.0,
        "transactions_today": 1
    })
    data = response.json()
    risk_text = " ".join(data["risk_factors"])
    assert "no risk" in risk_text.lower()


# ---------------------------------------------------------------------------
# Input validation — bad inputs should return 422, not 500
# ---------------------------------------------------------------------------

def test_predict_missing_field_returns_422():
    """
    If a required field is missing, FastAPI/pydantic should return 422
    (Unprocessable Entity) — not a 500 crash.
    422 = "I understood the request but the data is invalid"
    500 = "Something exploded internally" (never acceptable for bad input)
    """
    response = client.post("/predict", json={
        "amount": 1000.0,
        # time_of_day, distance_from_home_km, transactions_today are missing
    })
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    """
    If a field has the wrong type (string instead of number),
    pydantic should catch it and return 422 before it reaches the model.
    """
    response = client.post("/predict", json={
        "amount": "not_a_number",   # should be float
        "time_of_day": 14,
        "distance_from_home_km": 5.0,
        "transactions_today": 1
    })
    assert response.status_code == 422


def test_predict_empty_body_returns_422():
    """
    Sending an empty body should return 422, not crash the server.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------

def test_metrics_endpoint_returns_200():
    """
    The /metrics endpoint is scraped by Prometheus every 15 seconds.
    It must always be reachable and return 200.
    """
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_contains_prediction_counter():
    """
    After a prediction is made, the predictions_total counter
    must appear in the /metrics output.
    Prometheus won't graph anything if this metric is missing.
    """
    # Make a prediction first so the counter is non-zero
    client.post("/predict", json={
        "amount": 1000.0,
        "time_of_day": 10,
        "distance_from_home_km": 5.0,
        "transactions_today": 1
    })

    response = client.get("/metrics")
    assert "predictions_total" in response.text


def test_metrics_contains_latency_histogram():
    """
    The prediction latency histogram must appear in /metrics.
    This is what Grafana uses to show P50/P95/P99 latency graphs.
    """
    client.post("/predict", json={
        "amount": 1000.0,
        "time_of_day": 10,
        "distance_from_home_km": 5.0,
        "transactions_today": 1
    })

    response = client.get("/metrics")
    assert "prediction_latency_seconds" in response.text