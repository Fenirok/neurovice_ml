import os
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgb_adhd_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "zscore_scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_ORDER = [
    "adhd_composite",
    "inattention",
    "hyperactivity",
    "anxiety_index",
    "conduct_index",
    "odd_index"
]

def predict_risk(data: dict) -> float:
    features = np.array([[data[f] for f in FEATURE_ORDER]])
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]
    return round(float(prob), 4)
