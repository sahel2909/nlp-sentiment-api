"""
Text Classification using TF-IDF + Logistic Regression
Production Ready Version
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = Path("data/clean/cleaned_texts.csv")
MODEL_PATH = Path("models/sentiment_model.pkl")

LABEL_MAPPING = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}


def load_dataset() -> tuple[list[str], list[int]]:
    dataframe = pd.read_csv(DATA_PATH)
    texts = dataframe["clean_text"].astype(str).tolist()
    labels = dataframe["label"].astype(int).tolist()
    return texts, labels


def train_model(texts: list[str], labels: list[int]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    classifier = LogisticRegression(
        max_iter=2000,
        n_jobs=1
    )

    classifier.fit(x_train_vec, y_train)

    predictions = classifier.predict(x_test_vec)

    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    joblib.dump(
        {"model": classifier, "vectorizer": vectorizer},
        MODEL_PATH
    )

    print(f"\nModel saved at: {MODEL_PATH}")


class SentimentModel:
    """
    Load once, predict many times (API Ready)
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file not found. Train first.")

        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.vectorizer = bundle["vectorizer"]

    def predict(self, text: str) -> dict:
        vector = self.vectorizer.transform([text])
        label_id = int(self.model.predict(vector)[0])

        return {
            "text": text,
            "label_id": label_id,
            "label": LABEL_MAPPING[label_id]
        }


def main() -> None:
    texts, labels = load_dataset()
    train_model(texts, labels)

    model = SentimentModel()
    result = model.predict("battery backup is very poor")
    print("\nPrediction Result:", result)


if __name__ == "__main__":
    main()