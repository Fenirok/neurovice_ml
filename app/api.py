from fastapi import FastAPI
from inference.predict import predict_risk

app = FastAPI(title="ADHD Risk Prediction API")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    prob = predict_risk(data)

    return {
        "adhd_risk_score": prob,
    }
