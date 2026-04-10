"""
Text Clustering using KMeans + Sentence Embeddings (API Ready Version)
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Load model once
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)


def cluster_texts(texts: list[str], n_clusters: int = 3) -> dict:
    """
    Cluster texts into semantic groups.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    labels = kmeans.fit_predict(embeddings)

    clusters = {}

    for cluster_id in range(n_clusters):
        clusters[f"cluster_{cluster_id + 1}"] = [
            text for text, label in zip(texts, labels)
            if label == cluster_id
        ]

    score = silhouette_score(embeddings, labels)

    return {
        "silhouette_score": float(score),
        "clusters": clusters
    }


if __name__ == "__main__":
    df = pd.read_csv("data/clean/cleaned_texts.csv")
    print(cluster_texts(df["clean_text"].astype(str).tolist()))
