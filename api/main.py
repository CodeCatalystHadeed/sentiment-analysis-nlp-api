"""FastAPI application for sentiment prediction."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model
from src.preprocessing import preprocess_text

app = FastAPI(
    title="Sentiment Analysis NLP API",
    description="Predict positive, negative, or neutral sentiment from user text.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")


class PredictionResponse(BaseModel):
    sentiment: str


@lru_cache(maxsize=1)
def get_model():
    """Cache the trained model for repeated API calls."""
    return load_model()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
        cleaned_text = preprocess_text(request.text)
        sentiment = str(model.predict([cleaned_text])[0]).strip().lower()
        return PredictionResponse(sentiment=sentiment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
