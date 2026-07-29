"""Text preprocessing helpers."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "you",
    "your",
    "i",
    "have",
    "has",
    "had",
    "my",
    "we",
    "our",
    "me",
}


def preprocess_text(text: str) -> str:
    """Normalize text for classical NLP models."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in _STOPWORDS and len(token) > 1]
    return " ".join(tokens)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn compatible text normalizer."""

    def fit(self, X: Iterable[str], y: object = None) -> "TextPreprocessor":
        return self

    def transform(self, X: Iterable[str]) -> pd.Series:
        return pd.Series([preprocess_text(text) for text in X])
