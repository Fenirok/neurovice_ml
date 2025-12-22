import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/train11.csv")

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
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully")
