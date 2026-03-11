from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse
import time

app = FastAPI(title="Fraud Detection API")

# Load model
with open("model/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

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