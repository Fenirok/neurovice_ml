from fastapi import FastAPI
from app.prediction import predict_risk

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    prob = predict_risk(data)

    return {
        "adhd_risk_score": prob
    }
