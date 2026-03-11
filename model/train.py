import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate synthetic fraud data
X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=6,
    weights=[0.95, 0.05],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to model/fraud_model.pkl")