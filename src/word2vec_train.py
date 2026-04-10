# src/word2vec_train.py
"""
Train a Word2Vec model using cleaned text data.
"""

from __future__ import annotations
import os
import pandas as pd
from gensim.models import Word2Vec


def load_sentences(csv_path: str) -> list[list[str]]:
    df = pd.read_csv(csv_path)

    if "clean_text" not in df.columns:
        raise ValueError("Missing 'clean_text' column in CSV")

    return df["clean_text"].astype(str).str.split().tolist()


def train_word2vec(sentences: list[list[str]]) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=os.cpu_count(),
        sg=1,
        epochs=10
    )
    return model


def main() -> None:
    input_path = "data/clean/cleaned_texts.csv"
    output_path = "models/word2vec.model"

    os.makedirs("models", exist_ok=True)

    sentences = load_sentences(input_path)
    model = train_word2vec(sentences)

    model.save(output_path)
    print("Word2Vec model saved:", output_path)

    if "good" in model.wv and "excellent" in model.wv:
        sim = model.wv.similarity("good", "excellent")
        print("Similarity (good vs excellent):", round(sim, 3))


if __name__ == "__main__":
    main()
