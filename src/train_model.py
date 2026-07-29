"""Train and evaluate sentiment classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import preprocess_text
from src.utils import DATA_PATH, MODEL_PATH, ensure_directory, load_dataset


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    classification_report: str


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=preprocess_text, ngram_range=(1, 2), min_df=1)),
            ("classifier", estimator),
        ]
    )


def evaluate_model(name: str, pipeline: Pipeline, x_test: pd.Series, y_test: pd.Series) -> ModelResult:
    predictions = pipeline.predict(x_test)
    return ModelResult(
        name=name,
        pipeline=pipeline,
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, average="weighted", zero_division=0),
        recall=recall_score(y_test, predictions, average="weighted", zero_division=0),
        f1=f1_score(y_test, predictions, average="weighted", zero_division=0),
        confusion_matrix=confusion_matrix(y_test, predictions).tolist(),
        classification_report=classification_report(y_test, predictions, zero_division=0),
    )


def train_best_model(dataset_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict[str, ModelResult]:
    df = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    candidates = {
        "logistic_regression": build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "naive_bayes": build_pipeline(MultinomialNB()),
        "svm": build_pipeline(LinearSVC(class_weight="balanced")),
    }

    results: dict[str, ModelResult] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        results[name] = evaluate_model(name, pipeline, x_test, y_test)

    best_result = max(results.values(), key=lambda item: item.f1)
    ensure_directory(model_path.parent)
    joblib.dump(best_result.pipeline, model_path)
    return results


if __name__ == "__main__":
    outcomes = train_best_model()
    best = max(outcomes.values(), key=lambda item: item.f1)
    print(f"Best model: {best.name}")
    print(f"Accuracy: {best.accuracy:.4f}")
    print(f"F1-score: {best.f1:.4f}")
