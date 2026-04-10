"""
Module 4: Semantic Search using Sentence-BERT
Python 3.13
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "all-MiniLM-L6-v2"
DATA_PATH = Path("data/clean/cleaned_texts.csv")
EMBEDDING_PATH = Path("features/sentence_embeddings.npy")


def load_texts() -> list[str]:
    df = pd.read_csv(DATA_PATH)

    if "clean_text" not in df.columns:
        raise ValueError("Missing 'clean_text' column in CSV")

    texts = df["clean_text"].astype(str).tolist()

    if not texts:
        raise ValueError("No texts found in CSV")

    return texts


def build_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )


def semantic_search(
    model: SentenceTransformer,
    query: str,
    texts: list[str],
    embeddings: np.ndarray,
    top_k: int = 5
) -> None:

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = scores.argsort(descending=True)[:top_k]

    print("\nTop Semantic Matches:\n")

    for idx in top_results:
        idx = int(idx)
        print(f"Score: {scores[idx]:.4f}")
        print(f"Text: {texts[idx]}")
        print("-" * 50)


def main() -> None:

    texts = load_texts()

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    if not EMBEDDING_PATH.exists():
        EMBEDDING_PATH.parent.mkdir(exist_ok=True)
        embeddings = build_embeddings(model, texts)
        np.save(EMBEDDING_PATH, embeddings)
        print("[OK] Embeddings saved.")
    else:
        embeddings = np.load(EMBEDDING_PATH)
        print("[OK] Loaded existing embeddings.")

    semantic_search(
        model=model,
        query="battery performance is excellent",
        texts=texts,
        embeddings=embeddings,
        top_k=5
    )


if __name__ == "__main__":
    main()