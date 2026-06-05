import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="PCA Dimensionality Reduction",
    layout="wide"
)

st.title("Principal Component Analysis (PCA)")

# Load Dataset
wine = load_wine()

df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# Components
n_components = st.slider(
    "Select Number of Principal Components",
    min_value=2,
    max_value=min(df.shape[1],10),
    value=2
)

# PCA
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_scaled)

# Explained Variance
st.subheader("Explained Variance Ratio")

variance_df = pd.DataFrame({
    "Component":
    [f"PC{i+1}" for i in range(n_components)],
    "Variance":
    pca.explained_variance_ratio_
})

fig1 = px.bar(
    variance_df,
    x="Component",
    y="Variance",
    title="Variance Explained by Each Component"
)

st.plotly_chart(fig1, use_container_width=True)

# 2D PCA Plot
if n_components >= 2:

    pca_2d = PCA(n_components=2)

    X_vis = pca_2d.fit_transform(X_scaled)

    plot_df = pd.DataFrame({
        "PC1": X_vis[:,0],
        "PC2": X_vis[:,1]
    })

    fig2 = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        title="PCA Projection"
    )

    st.plotly_chart(fig2, use_container_width=True)

# Reduced Dataset
reduced_df = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

st.subheader("Reduced Dataset")
st.dataframe(reduced_df.head())

csv = reduced_df.to_csv(index=False)

st.download_button(
    "Download PCA Dataset",
    csv,
    "pca_reduced_data.csv",
    "text/csv"
)