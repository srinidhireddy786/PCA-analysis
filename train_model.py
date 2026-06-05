from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import os
import pandas as pd

# Load Iris Dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# DBSCAN
dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

dbscan.fit(X_scaled)

# Save scaler
os.makedirs("models", exist_ok=True)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Training Complete")