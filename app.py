import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

st.set_page_config(
    page_title="DBSCAN Clustering - Iris Dataset",
    layout="wide"
)

st.title("DBSCAN Clustering on Iris Dataset")

# Load Dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

st.subheader("Dataset Preview")

st.dataframe(df.head())

# Parameters
eps = st.slider(
    "EPS",
    0.1,
    2.0,
    0.8,
    0.1
)

min_samples = st.slider(
    "Min Samples",
    2,
    15,
    5
)

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# DBSCAN
model = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

df["Cluster"] = model.fit_predict(X_scaled)

# Scatter Plot
fig = px.scatter(
    df,
    x="petal length (cm)",
    y="petal width (cm)",
    color=df["Cluster"].astype(str),
    title="DBSCAN Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Cluster Distribution
cluster_counts = (
    df["Cluster"]
    .value_counts()
    .reset_index()
)

cluster_counts.columns = [
    "Cluster",
    "Count"
]

fig2 = px.bar(
    cluster_counts,
    x="Cluster",
    y="Count",
    title="Cluster Distribution"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# Noise Points
noise = (
    df["Cluster"] == -1
).sum()

st.metric(
    "Noise Points",
    noise
)

# Cluster Summary
st.subheader("Cluster Summary")

st.dataframe(
    df.groupby("Cluster").mean()
)

# Download Results
csv = df.to_csv(index=False)

st.download_button(
    "Download Clustered Dataset",
    csv,
    "iris_dbscan_clusters.csv",
    "text/csv"
)