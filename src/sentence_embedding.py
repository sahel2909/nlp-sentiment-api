# src/sentence_embedding.py
"""
Generate sentence embeddings using Sentence-BERT.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


def load_sentences(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)

    if "clean_text" not in df.columns:
        raise ValueError("Missing 'clean_text' column in CSV")

    sentences = df["clean_text"].astype(str).tolist()

    if len(sentences) == 0:
        raise ValueError("No sentences found in CSV")

    return sentences


def sentences_similarity(s1: str, s2: str, model: SentenceTransformer) -> float:
    embeddings = model.encode([s1, s2], normalize_embeddings=True)
    score = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    return float(score)


def main() -> None:
    input_path = "data/clean/cleaned_texts.csv"
    output_path = "features/sentence_embeddings.npy"

    os.makedirs("features", exist_ok=True)

    sentences = load_sentences(input_path)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    embeddings = model.encode(sentences, normalize_embeddings=True)

    np.save(output_path, embeddings)
    print("Embeddings saved to:", output_path)
    print("Embedding shape:", embeddings.shape)

    sim = sentences_similarity(
        "battery life is very good",
        "the phone lasts long without charging",
        model
    )

    print("Similarity score:", round(sim, 3))


if __name__ == "__main__":
    main()
