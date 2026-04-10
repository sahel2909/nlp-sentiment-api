"""
Production Ready NLP API
Sentence-BERT Sentiment Classification + Semantic Search
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util


# -----------------------
# Config
# -----------------------

MODEL_PATH = Path("models/sbert_sentiment_model.pkl")


# -----------------------
# FastAPI App
# -----------------------

app = FastAPI(
    title="NLP API",
    description="Sentiment Classification & Semantic Search API",
    version="2.0.0"
)


# -----------------------
# Request Schemas
# -----------------------

class TextInput(BaseModel):
    """Input text for sentiment prediction"""
    text: str = Field(..., min_length=1)


class SearchInput(BaseModel):
    """Input for semantic search"""
    query: str = Field(..., min_length=1)
    corpus: list[str]
    top_k: int = 5


# -----------------------
# Load Models
# -----------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError("❌ Sentiment model not found. Train model first.")

classifier = joblib.load(MODEL_PATH)

# Auto device selection (CPU/GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=DEVICE
)

LABEL_MAPPING = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"
}


# -----------------------
# Routes
# -----------------------

@app.get("/")
def health_check():
    return {"status": "API is running successfully"}


@app.post("/predict")
def predict_sentiment(data: TextInput):
    try:
        embedding = embedder.encode(
            [data.text],
            normalize_embeddings=True
        )

        label = int(classifier.predict(embedding)[0])

        sentiment = LABEL_MAPPING.get(label, "Unknown")

        return {
            "text": data.text,
            "label_id": label,
            "sentiment": sentiment
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/semantic-search")
def semantic_search(data: SearchInput):
    try:
        if not data.corpus:
            raise HTTPException(
                status_code=400,
                detail="Corpus cannot be empty."
            )

        corpus_embeddings = embedder.encode(
            data.corpus,
            normalize_embeddings=True
        )

        query_embedding = embedder.encode(
            [data.query],
            normalize_embeddings=True
        )

        scores = util.cos_sim(
            query_embedding,
            corpus_embeddings
        )[0].cpu().numpy()

        ranked_indices = np.argsort(scores)[::-1]

        top_k = min(data.top_k, len(data.corpus))

        results = [
            {
                "text": data.corpus[index],
                "score": float(scores[index])
            }
            for index in ranked_indices[:top_k]
        ]

        return {
            "query": data.query,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e