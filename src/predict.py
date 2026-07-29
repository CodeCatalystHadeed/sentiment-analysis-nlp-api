"""Prediction helpers for the sentiment analysis API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.utils import MODEL_PATH, normalize_sentiment_label


@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load and cache the trained model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run src/train_model.py first."
        )
    return joblib.load(model_path)


def predict_sentiment(text: str, model_path: Path = MODEL_PATH) -> str:
    """Predict the sentiment label for a single text input."""
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty.")
    model = load_model(model_path)
    prediction = model.predict([text])[0]
    return normalize_sentiment_label(str(prediction))
