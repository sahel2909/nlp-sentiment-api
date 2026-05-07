from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="NLP API",
    version="1.0.0"
)

class TextInput(BaseModel):
    text: str = Field(..., min_length=1)

class SearchInput(BaseModel):
    query: str
    corpus: list[str]
    top_k: int = 5


@app.get("/")
def home():
    return {"status": "API is running successfully 🚀"}


@app.post("/predict")
def predict_sentiment(data: TextInput):
    # dummy response (temporary)
    return {
        "text": data.text,
        "sentiment": "Positive"
    }


@app.post("/semantic-search")
def semantic_search(data: SearchInput):
    results = [
        {"text": t, "score": 0.99}
        for t in data.corpus[:data.top_k]
    ]

    return {
        "query": data.query,
        "results": results
    }