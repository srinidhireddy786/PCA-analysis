import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

st.set_page_config(
    page_title="t-SNE Visualization",
    layout="wide"
)

st.title("t-SNE Dimensionality Reduction")

digits = load_digits()

df = pd.DataFrame(
    digits.data
)

target = digits.target

st.subheader("Dataset Shape")

st.write(df.shape)

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

perplexity = st.slider(
    "Perplexity",
    min_value=5,
    max_value=50,
    value=30
)

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "TSNE1": X_tsne[:,0],
    "TSNE2": X_tsne[:,1],
    "Digit": target.astype(str)
})

fig = px.scatter(
    plot_df,
    x="TSNE1",
    y="TSNE2",
    color="Digit",
    title="t-SNE Visualization of Digits Dataset"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Reduced Dataset")

st.dataframe(
    plot_df.head()
)

csv = plot_df.to_csv(index=False)

st.download_button(
    "Download t-SNE Dataset",
    csv,
    "tsne_reduced_data.csv",
    "text/csv"
)