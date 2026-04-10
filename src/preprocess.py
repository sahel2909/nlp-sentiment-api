"""
Text Preprocessing
Production-safe preprocessing for NLP pipelines.
"""

import re
import sys
import pandas as pd


def clean_text(text: str) -> str:
    """
    Clean raw text.

    Steps:
    - Lowercase
    - Remove URLs
    - Remove emails
    - Remove special characters
    - Normalize spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_series(series: pd.Series) -> pd.Series:
    """
    Apply text cleaning to a pandas Series.
    """
    return series.fillna("").apply(clean_text)


def main(input_path: str, output_path: str) -> None:
    """
    Read CSV, preprocess 'text' column, save cleaned CSV.
    """
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column")

    df["clean_text"] = preprocess_series(df["text"])
    df.to_csv(output_path, index=False)

    print(f"[OK] Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/preprocess.py input.csv output.csv")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
