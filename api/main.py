"""FastAPI application for sentiment prediction."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_sentiment

app = FastAPI(
    title="Sentiment Analysis NLP API",
    description="Predict positive, negative, or neutral sentiment from user text.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to classify")


class PredictionResponse(BaseModel):
    sentiment: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        sentiment = predict_sentiment(request.text)
        return PredictionResponse(sentiment=sentiment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
