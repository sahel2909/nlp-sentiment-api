"""
----------------------------------
This script converts cleaned text into TF-IDF features.

Output:
- features/tfidf_features.pkl
- features/tfidf_vectorizer.pkl
"""

import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


FEATURE_DIR = Path("features")


def extract_tfidf(input_path: str) -> None:
    """
    Extract TF-IDF features from cleaned text CSV.

    Args:
        input_path (str): Path to cleaned CSV file.
    """
    FEATURE_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean_text' column")

    texts = df["clean_text"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        max_features=5000,          # Top 5,000 features
        ngram_range=(1, 2),         # Unigrams + bigrams
        stop_words="english"
    )

    features = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, FEATURE_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(features, FEATURE_DIR / "tfidf_features.pkl")

    print(
        f"[OK] TF-IDF extracted: "
        f"{features.shape[0]} samples, {features.shape[1]} features"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/features_tfidf.py <cleaned_csv_path>")
        sys.exit(1)

    extract_tfidf(sys.argv[1])