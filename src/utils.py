"""Shared utilities for the sentiment analysis project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "reviews.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the sentiment dataset from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add a CSV with columns: text,sentiment."
        )
    df = pd.read_csv(path)
    expected_columns = {"text", "sentiment"}
    if not expected_columns.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'sentiment' columns.")
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["text"] = df["text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    return df


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_sentiment_label(label: str) -> str:
    """Normalize the model label to the API response format."""
    return label.strip().lower()


def flatten(iterable: Iterable[Iterable[str]]) -> list[str]:
    """Flatten a nested iterable of strings."""
    return [item for group in iterable for item in group]
