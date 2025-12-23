import os
import numpy as np
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "xgb_adhd_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "zscore_scaler.pkl")

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
    """
    Expects raw (non-normalized) feature values.
    Returns ADHD risk probability in 0–1 range.
    """

    # ✅ Create DataFrame with feature names (FIX)
    df = pd.DataFrame([{
        "adhd_composite": data["adhd_composite"],
        "inattention": data["inattention"],
        "hyperactivity": data["hyperactivity"],
        "anxiety_index": data["anxiety_index"],
        "conduct_index": data["conduct_index"],
        "odd_index": data["odd_index"],
    }])

    # Scale using trained scaler
    scaled = scaler.transform(df)

    # Predict probability for positive class (ADHD = 1)
    prob = model.predict_proba(scaled)[0][1]

    return round(float(prob), 4)