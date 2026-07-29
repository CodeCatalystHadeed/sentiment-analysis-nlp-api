from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from api.main import app


client = TestClient(app)


class DummyModel:
    def predict(self, texts):
        return ["positive"]


def test_valid_prediction_request(monkeypatch):
    monkeypatch.setattr("src.predict.load_model", lambda model_path=None: DummyModel())
    response = client.post("/predict", json={"text": "I love this product"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"


def test_empty_text_handling():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_invalid_input_handling():
    response = client.post("/predict", json={"wrong_field": "text"})
    assert response.status_code == 422
