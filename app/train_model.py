import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "train11.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load data
df = pd.read_csv(DATA_PATH)

X = df.drop("adhd_label", axis=1)
y = df["adhd_label"]

# Z-score using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss"
)
model.fit(X_scaled, y)

# Save model & scaler
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully")
