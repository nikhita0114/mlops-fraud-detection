from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time
import os

app = FastAPI(
    title="Fraud Detection API",
    description="Detects fraudulent transactions using real-world inputs"
)

# Train model if it doesn't exist
def get_model():
    model_path = "model/fraud_model.pkl"
    if not os.path.exists(model_path):
        print("Model not found, training now...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(
            n_samples=10000, n_features=10, n_informative=6,
            weights=[0.7, 0.3], random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42
        )
        model.fit(X_train, y_train)
        os.makedirs("model", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("✅ Model trained and saved!")

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = get_model()

# Prometheus metrics
PREDICTIONS = Counter("predictions_total", "Total predictions", ["result"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

# Real-world input model
class Transaction(BaseModel):
    amount: float               # Transaction amount in ₹
    time_of_day: int            # Hour of day (0-23)
    distance_from_home_km: float  # Distance from home in km
    transactions_today: int     # Number of transactions today

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 45000.00,
                "time_of_day": 2,
                "distance_from_home_km": 250.5,
                "transactions_today": 8
            }
        }

def preprocess(transaction: Transaction) -> np.ndarray:
    """Convert real-world inputs into model features"""

    amount_normalized = transaction.amount / 100000.0
    time_normalized = transaction.time_of_day / 23.0
    distance_normalized = transaction.distance_from_home_km / 1000.0
    txn_normalized = transaction.transactions_today / 20.0

    # Risk signals
    is_night = 1.0 if transaction.time_of_day < 6 or transaction.time_of_day > 22 else 0.0
    is_high_amount = 1.0 if transaction.amount > 50000 else 0.0
    is_far_from_home = 1.0 if transaction.distance_from_home_km > 100 else 0.0
    is_many_txns = 1.0 if transaction.transactions_today > 5 else 0.0
    combined_risk = (is_night + is_high_amount + is_far_from_home + is_many_txns) / 4.0
    unusual_time_amount = amount_normalized * is_night

    return np.array([[
        amount_normalized,
        time_normalized,
        distance_normalized,
        txn_normalized,
        is_night,
        is_high_amount,
        is_far_from_home,
        is_many_txns,
        combined_risk,
        unusual_time_amount
    ]])

def risk_explanation(transaction: Transaction) -> list:
    """Explain why a transaction is risky"""
    reasons = []
    if transaction.time_of_day < 6 or transaction.time_of_day > 22:
        reasons.append("⚠️ Transaction at unusual hour")
    if transaction.amount > 50000:
        reasons.append("⚠️ High transaction amount")
    if transaction.distance_from_home_km > 100:
        reasons.append("⚠️ Far from home location")
    if transaction.transactions_today > 5:
        reasons.append("⚠️ Many transactions today")
    return reasons if reasons else ["✅ No risk factors detected"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    start = time.time()

    features = preprocess(transaction)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = "fraud" if prediction == 1 else "legitimate"
    PREDICTIONS.labels(result=result).inc()
    LATENCY.observe(time.time() - start)

    return {
        "prediction": result,
        "fraud_probability": f"{round(float(probability) * 100, 1)}%",
        "risk_factors": risk_explanation(transaction),
        "transaction_summary": {
            "amount": f"₹{transaction.amount:,.2f}",
            "time": f"{transaction.time_of_day}:00 hrs",
            "distance_from_home": f"{transaction.distance_from_home_km} km",
            "transactions_today": transaction.transactions_today
        }
    }

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)