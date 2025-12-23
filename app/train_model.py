import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# BASE DIR = neurovice_ml/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "final_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "xgb_adhd_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "zscore_scaler.pkl")

# Load data
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "adhd_composite",
    "inattention",
    "hyperactivity",
    "anxiety_index",
    "conduct_index",
    "odd_index"
]

X = df[FEATURES]
y = df["adhd_label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Z-score scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Save artifacts in ROOT
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Model & scaler saved at project root")
