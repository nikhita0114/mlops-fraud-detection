from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time
import os

app = FastAPI(title="Fraud Detection API")

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
            weights=[0.95, 0.05], random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
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

class Transaction(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    start = time.time()
    features = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    result = "fraud" if prediction == 1 else "legitimate"
    PREDICTIONS.labels(result=result).inc()
    LATENCY.observe(time.time() - start)
    return {
        "prediction": result,
        "fraud_probability": round(float(probability), 4)
    }

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
