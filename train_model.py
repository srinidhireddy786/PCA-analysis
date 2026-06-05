from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd

digits = load_digits()

df = pd.DataFrame(digits.data)

scaler = StandardScaler()

scaler.fit(df)

os.makedirs("models", exist_ok=True)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler Saved Successfully")