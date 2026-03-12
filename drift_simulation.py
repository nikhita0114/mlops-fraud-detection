import requests
import numpy as np
import time

API_URL = "http://localhost:8000/predict"

def send_transaction(amount, hour, distance, txns):
    response = requests.post(API_URL, json={
        "amount": amount,
        "time_of_day": hour,
        "distance_from_home_km": distance,
        "transactions_today": txns
    })
    return response.json()

print("=" * 60)
print("DRIFT SIMULATION — Fraud Detection System")
print("=" * 60)

# Phase 1 — Normal traffic (low fraud)
print("\n📊 PHASE 1: Normal Traffic (legitimate customers)")
print("-" * 40)
fraud_count = 0
total = 20
for i in range(total):
    result = send_transaction(
        amount=np.random.uniform(100, 5000),
        hour=np.random.randint(9, 21),
        distance=np.random.uniform(1, 20),
        txns=np.random.randint(1, 3)
    )
    if result["prediction"] == "fraud":
        fraud_count += 1
    time.sleep(0.3)

print(f"Fraud detected: {fraud_count}/{total} ({fraud_count/total*100:.1f}%)")

# Phase 2 — Drift begins (unusual patterns)
print("\n⚠️  PHASE 2: Drift Begins (unusual patterns emerging)")
print("-" * 40)
fraud_count = 0
for i in range(total):
    result = send_transaction(
        amount=np.random.uniform(10000, 40000),
        hour=np.random.randint(0, 6),
        distance=np.random.uniform(50, 150),
        txns=np.random.randint(4, 7)
    )
    if result["prediction"] == "fraud":
        fraud_count += 1
    time.sleep(0.3)

print(f"Fraud detected: {fraud_count}/{total} ({fraud_count/total*100:.1f}%)")

# Phase 3 — Full drift (attack pattern)
print("\n🚨 PHASE 3: Full Drift — ATTACK PATTERN DETECTED!")
print("-" * 40)
fraud_count = 0
for i in range(total):
    result = send_transaction(
        amount=np.random.uniform(50000, 100000),
        hour=np.random.randint(0, 4),
        distance=np.random.uniform(200, 500),
        txns=np.random.randint(8, 15)
    )
    if result["prediction"] == "fraud":
        fraud_count += 1
    time.sleep(0.3)

print(f"Fraud detected: {fraud_count}/{total} ({fraud_count/total*100:.1f}%)")

print("\n" + "=" * 60)
print("DRIFT SIMULATION COMPLETE")
print("Check Grafana at http://localhost:3000 to see the spike!")
print("=" * 60)
