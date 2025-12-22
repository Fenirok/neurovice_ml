from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load trained model ONLY
model = pickle.load(open("model.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    """
    Expects PRE-NORMALIZED features.
    Feature order MUST match training order.
    """

    features = np.array([[
        data["adhd_composite"],
        data["inattention"],
        data["hyperactivity"],
        data["anxiety_index"],
        data["conduct_index"],
        data["odd_index"],
    ]])

    prob = float(model.predict_proba(features)[0][1])

    return {
        "adhd_risk": round(prob, 4)  # 0–1 range
    }
