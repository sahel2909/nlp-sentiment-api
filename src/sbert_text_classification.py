"""
Sentence-BERT based Text Classification
Production Ready Version (CPU)
"""

from pathlib import Path
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = Path("data/clean/cleaned_texts.csv")
MODEL_PATH = Path("models/sbert_sentiment_model.pkl")

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


def generate_embeddings(texts: list[str], model: SentenceTransformer):
    return model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
    )


def train_classifier(texts: list[str], labels: list[int]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )

    embeddings = generate_embeddings(texts, embedding_model)

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    classifier = LogisticRegression(
        max_iter=3000,
        n_jobs=1
    )

    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)

    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    joblib.dump(classifier, MODEL_PATH)

    print(f"\nModel saved at: {MODEL_PATH}")


class SBERTSentimentModel:
    """
    Load once, predict many times (API Ready)
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file not found. Train first.")

        self.classifier = joblib.load(MODEL_PATH)
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

    def predict(self, text: str) -> dict:
        embedding = self.embedding_model.encode([text])
        label_id = int(self.classifier.predict(embedding)[0])

        return {
            "text": text,
            "label_id": label_id,
            "label": LABEL_MAPPING[label_id]
        }


def main() -> None:
    texts, labels = load_dataset()
    train_classifier(texts, labels)

    model = SBERTSentimentModel()
    result = model.predict("camera quality is fantastic")
    print("\nPrediction Result:", result)


if __name__ == "__main__":
    main()