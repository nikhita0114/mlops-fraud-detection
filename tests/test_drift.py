"""
Tests for the drift detection endpoint.

We test:
- /drift-report returns correct structure
- Returns "not enough data" message with < 10 samples
- Detects no drift for normal transactions (matches reference distribution)
- Detects drift for anomalous transactions (Phase 3 attack pattern)
"""

from fastapi.testclient import TestClient
from app.main import app, PREDICTION_WINDOW

client = TestClient(app)


def clear_window():
    """Helper to reset the prediction window between tests."""
    PREDICTION_WINDOW.clear()


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_drift_report_returns_200():
    """The /drift-report endpoint must always return 200."""
    clear_window()
    response = client.get("/drift-report")
    assert response.status_code == 200


def test_drift_report_returns_required_fields():
    """Response must contain all required fields."""
    clear_window()
    response = client.get("/drift-report")
    data = response.json()
    assert "drift_detected" in data
    assert "drifted_features" in data
    assert "share_drifted" in data
    assert "current_window_size" in data


def test_drift_report_not_enough_data():
    """
    With fewer than 10 predictions, drift report should return a message
    explaining there's not enough data — not crash or return misleading results.
    """
    clear_window()
    # Make only 5 predictions
    for _ in range(5):
        client.post("/predict", json={
            "amount": 500.0,
            "time_of_day": 14,
            "distance_from_home_km": 5.0,
            "transactions_today": 1
        })

    response = client.get("/drift-report")
    assert response.status_code == 200
    data = response.json()
    assert data["drift_detected"] is False
    assert data["current_window_size"] == 5
    assert "message" in data


# ---------------------------------------------------------------------------
# Drift detection accuracy tests
# ---------------------------------------------------------------------------

def test_no_drift_for_normal_transactions():
    """
    Normal transactions (matching the reference distribution) should
    NOT trigger drift detection.
    """
    clear_window()
    # Send 30 normal transactions — small amounts, business hours, nearby
    for i in range(30):
        client.post("/predict", json={
            "amount": float(500 + (i * 100)),   # ₹500 - ₹3,500
            "time_of_day": 9 + (i % 12),        # 9am - 9pm
            "distance_from_home_km": float(2 + i),  # 2-32km
            "transactions_today": 1 + (i % 3)   # 1-3 transactions
        })

    response = client.get("/drift-report")
    data = response.json()
    assert data["current_window_size"] == 30
    # Normal transactions should not trigger drift
    assert data["drift_detected"] is False


def test_drift_detected_for_attack_pattern():
    """
    Phase 3 attack pattern (huge amounts, 3am, 400km away, 10+ txns)
    should trigger drift detection — these are completely outside the
    reference distribution.
    """
    clear_window()
    # Send 30 attack-pattern transactions
    for _ in range(30):
        client.post("/predict", json={
            "amount": 95000.0,       # far above reference range
            "time_of_day": 3,        # 3am — outside reference hours
            "distance_from_home_km": 450.0,  # far outside reference range
            "transactions_today": 12          # far above reference range
        })

    response = client.get("/drift-report")
    data = response.json()
    assert data["current_window_size"] == 30
    # Attack pattern should trigger drift
    assert data["drift_detected"] is True
    assert len(data["drifted_features"]) > 0


def test_drift_report_per_feature_structure():
    """
    When enough data exists, per_feature should contain details
    for each of the 4 input features.
    """
    clear_window()
    for _ in range(20):
        client.post("/predict", json={
            "amount": 95000.0,
            "time_of_day": 3,
            "distance_from_home_km": 450.0,
            "transactions_today": 12
        })

    response = client.get("/drift-report")
    data = response.json()
    per_feature = data["per_feature"]

    # All 4 features should have drift details
    for feature in ["amount", "time_of_day", "distance_from_home_km", "transactions_today"]:
        assert feature in per_feature
        assert "drift_detected" in per_feature[feature]
        assert "drift_score" in per_feature[feature]


def test_window_accumulates_across_predictions():
    """
    Each call to /predict should add to the window.
    The window size should match the number of predictions made.
    """
    clear_window()
    for i in range(15):
        client.post("/predict", json={
            "amount": 1000.0,
            "time_of_day": 12,
            "distance_from_home_km": 10.0,
            "transactions_today": 1
        })

    response = client.get("/drift-report")
    data = response.json()
    assert data["current_window_size"] == 15