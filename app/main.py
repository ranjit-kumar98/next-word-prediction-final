from fastapi import FastAPI
from pydantic import BaseModel
from app.model import model, word_to_index, index_to_word, SEQUENCE_LENGTH
from app.utils import predict_next_words

app = FastAPI(title="Next Word Prediction API")

class PredictionRequest(BaseModel):
    text: str
    top_k: int = 5
    temperature: float = 1.0

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(req: PredictionRequest):
    predictions = predict_next_words(
        model,
        req.text,
        word_to_index,
        index_to_word,
        SEQUENCE_LENGTH,
        req.top_k,
        req.temperature
    )
    return {
        "input": req.text,
        "predictions": predictions
    }
