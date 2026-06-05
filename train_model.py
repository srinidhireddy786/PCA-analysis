from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd

wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

scaler = StandardScaler()

scaler.fit(df)

os.makedirs("models", exist_ok=True)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler Saved Successfully")