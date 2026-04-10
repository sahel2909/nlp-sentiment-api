"""
-------------------------------------------------
Output:
- models/tfidf_logreg_model.pkl
"""

import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODEL_DIR = Path("models")
FEATURE_DIR = Path("features")


def train_model(input_path: str) -> None:
    """
    Train Logistic Regression model using TF-IDF features.

    Args:
        input_path (str): Path to cleaned CSV file.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    FEATURE_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(input_path).dropna()

    required_columns = {"clean_text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV must contain 'clean_text' and 'label' columns")

    # Load vectorizer
    vectorizer_path = FEATURE_DIR / "tfidf_vectorizer.pkl"
    if not vectorizer_path.exists():
        raise FileNotFoundError("TF-IDF vectorizer not found. Run features_tfidf.py first.")

    vectorizer = joblib.load(vectorizer_path)

    x = vectorizer.transform(df["clean_text"].astype(str))
    y = df["label"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[OK] Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    model_path = MODEL_DIR / "tfidf_logreg_model.pkl"
    joblib.dump(model, model_path)

    print(f"[OK] Model saved to: {model_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train_model_tfidf.py <cleaned_csv_path>")
        sys.exit(1)

    train_model(sys.argv[1])