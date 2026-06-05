import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

st.set_page_config(
    page_title="Wine Clustering using GMM",
    layout="wide"
)

st.title("Gaussian Mixture Model Clustering")

# Load Dataset
wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

st.subheader("Dataset Preview")

st.dataframe(df.head())

# Number of Components
n_components = st.slider(
    "Number of Components",
    2,
    10,
    3
)

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# GMM
gmm = GaussianMixture(
    n_components=n_components,
    random_state=42
)

clusters = gmm.fit_predict(X_scaled)

df["Cluster"] = clusters

# PCA for Visualization
pca = PCA(n_components=2)

pca_result = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame(
    {
        "PCA1": pca_result[:,0],
        "PCA2": pca_result[:,1],
        "Cluster": clusters
    }
)

# Scatter Plot
fig = px.scatter(
    plot_df,
    x="PCA1",
    y="PCA2",
    color=plot_df["Cluster"].astype(str),
    title="GMM Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# Cluster Distribution
cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
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

# Probability Matrix
st.subheader(
    "Cluster Membership Probabilities"
)

probs = gmm.predict_proba(X_scaled)

st.dataframe(
    pd.DataFrame(probs).head()
)

# Cluster Summary
st.subheader("Cluster Summary")

st.dataframe(
    df.groupby("Cluster").mean()
)

# Download
csv = df.to_csv(index=False)

st.download_button(
    "Download Clustered Dataset",
    csv,
    "wine_gmm_clusters.csv",
    "text/csv"
)