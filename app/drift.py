"""
Drift detection using Evidently.

This module compares recent production transactions (current window)
against the training data distribution (reference) to detect data drift.

Why isolate this in its own file?
- Keeps main.py focused on request handling
- Easier to test drift logic independently
- Can swap Evidently for another library without touching the API
"""

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Feature names — must match the order used in preprocess() in main.py
FEATURE_NAMES = [
    "amount",
    "time_of_day",
    "distance_from_home_km",
    "transactions_today",
]


def generate_reference_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate reference data representing the training distribution.

    In a real system this would load actual training data from a file or database.
    Here we generate synthetic data matching the same distribution used in train.py
    so the comparison is meaningful.

    The training distribution (normal legitimate transactions):
    - amount:              ₹100 - ₹10,000 (log-normal, most are small)
    - time_of_day:         business hours weighted (9am-9pm)
    - distance_from_home:  0-50km (most people shop nearby)
    - transactions_today:  1-3 (most people make few transactions)
    """
    rng = np.random.default_rng(42)  # fixed seed = reproducible reference

    reference = pd.DataFrame({
        "amount": rng.lognormal(mean=7.0, sigma=1.0, size=n_samples),
        "time_of_day": rng.integers(9, 21, size=n_samples),
        "distance_from_home_km": rng.exponential(scale=15.0, size=n_samples),
        "transactions_today": rng.integers(1, 4, size=n_samples),
    })

    return reference


def run_drift_report(current_window: list[dict]) -> dict:
    """
    Compare current production transactions against the reference distribution.

    Args:
        current_window: list of recent transaction dicts, each with keys
                        matching FEATURE_NAMES

    Returns:
        dict with:
          - drift_detected:        bool, True if dataset-level drift detected
          - drifted_features:      list of feature names that drifted
          - share_drifted:         float, fraction of features that drifted
          - per_feature:           dict, per-feature drift details
          - current_window_size:   int, number of transactions analyzed
    """
    if len(current_window) < 10:
        # Not enough data for a meaningful statistical test
        return {
            "drift_detected": False,
            "drifted_features": [],
            "share_drifted": 0.0,
            "per_feature": {},
            "current_window_size": len(current_window),
            "message": f"Need at least 10 samples, have {len(current_window)}"
        }

    # Build DataFrames — Evidently needs pandas DataFrames
    reference_df = generate_reference_data()
    current_df = pd.DataFrame(current_window)[FEATURE_NAMES]

    # Build the Evidently report
    # DataDriftPreset runs the full suite of drift tests automatically
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    # Extract results from the report
    result = report.as_dict()

    # Parse Evidently's output structure
    drift_results = {}
    drifted_features = []

    try:
        metrics = result["metrics"]
        for metric in metrics:
            if metric["metric"] == "DatasetDriftMetric":
                dataset_drift = metric["result"]
                overall_drift = dataset_drift.get("dataset_drift", False)
                share_drifted = dataset_drift.get("share_of_drifted_columns", 0.0)

            if metric["metric"] == "DataDriftTable":
                columns = metric["result"].get("drift_by_columns", {})
                for feature, details in columns.items():
                    if feature in FEATURE_NAMES:
                        drifted = details.get("drift_detected", False)
                        drift_results[feature] = {
                            "drift_detected": drifted,
                            "drift_score": round(details.get("drift_score", 0.0), 4),
                            "stattest": details.get("stattest_name", "unknown"),
                        }
                        if drifted:
                            drifted_features.append(feature)

    except (KeyError, TypeError):
        overall_drift = False
        share_drifted = 0.0

    return {
        "drift_detected": overall_drift,
        "drifted_features": drifted_features,
        "share_drifted": round(share_drifted, 3),
        "per_feature": drift_results,
        "current_window_size": len(current_window),
    }