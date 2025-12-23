import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "final_data.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

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
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

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

joblib.dump(model, os.path.join(ARTIFACTS_DIR, "xgb_adhd_model.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "zscore_scaler.pkl"))

print(" Training complete, artifacts saved")
