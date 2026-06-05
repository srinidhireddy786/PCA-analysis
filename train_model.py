from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import pandas as pd
import joblib
import os

# Load Dataset
wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# GMM Model
gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

gmm.fit(X_scaled)

# Save
os.makedirs("models", exist_ok=True)

joblib.dump(
    gmm,
    "models/gmm_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Model Saved Successfully")