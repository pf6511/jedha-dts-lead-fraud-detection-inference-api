"""API tests without MLflow: the model loading is replaced by a fake model."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import fraud_detection_inference_api.main as main
from fraud_detection_contracts import FEATURE_SET

API_KEY = "test-api-key"


class FakeModel:
    """Returns a fixed fraud probability and records the columns received."""

    def __init__(self, probability: float):
        self.probability = probability
        self.received_columns: list[str] = []

    def predict_proba(self, X):
        self.received_columns = list(X.columns)
        return np.array([[1 - self.probability, self.probability]])


def payload() -> dict:
    return {
        "transaction": {
            "amt": 258.37,
            "city_pop": 725,
            "merchant": "fraud_Skiles LLC",
            "category": "home",
            "state": "IL",
            "job": "Environmental consultant",
        },
        "card_features": {
            "distance_km": 105.6,
            "card_transaction_count": 12,
            "card_fraud_count": 1,
            "card_fraud_rate": 0.0854,
        },
    }


def make_client(monkeypatch, probability: float = 0.6, threshold: float = 0.35):
    fake = FakeModel(probability)
    loaded = main.LoadedModel(
        model=fake,
        model_name="fraud-detection-random-forest",
        model_version="8",
        run_id="run123",
        threshold=threshold,
        feature_set=FEATURE_SET,
        feature_names=main.contract_feature_names(),
    )
    monkeypatch.setenv("INFERENCE_API_KEY", API_KEY)
    monkeypatch.setattr(main, "load_production_model", lambda: loaded)
    return TestClient(main.app), fake


def test_predict_returns_prediction_and_model_version(monkeypatch):
    client, fake = make_client(monkeypatch, probability=0.6, threshold=0.35)
    with client:
        response = client.post("/predict", json=payload(), headers={"X-API-Key": API_KEY})

    assert response.status_code == 200
    body = response.json()
    assert body["fraud_prediction"] is True
    assert body["fraud_probability"] == pytest.approx(0.6)
    assert body["model_version"] == "8"
    assert body["feature_set"] == FEATURE_SET
    assert set(fake.received_columns) == set(main.contract_feature_names())


def test_below_threshold_is_not_fraud(monkeypatch):
    client, _ = make_client(monkeypatch, probability=0.2, threshold=0.35)
    with client:
        response = client.post("/predict", json=payload(), headers={"X-API-Key": API_KEY})
    assert response.json()["fraud_prediction"] is False


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key"}])
def test_missing_or_wrong_api_key_is_rejected(monkeypatch, headers):
    client, _ = make_client(monkeypatch)
    with client:
        response = client.post("/predict", json=payload(), headers=headers)
    assert response.status_code == 401


def test_personal_field_is_rejected(monkeypatch):
    client, _ = make_client(monkeypatch)
    data = payload()
    data["transaction"]["cc_num"] = "2291163933867244"
    with client:
        response = client.post("/predict", json=data, headers={"X-API-Key": API_KEY})
    assert response.status_code == 422


def test_health_is_public_and_shows_model(monkeypatch):
    client, _ = make_client(monkeypatch)
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "8"


def test_incompatible_feature_set_is_refused():
    with pytest.raises(RuntimeError, match="does not match"):
        main.check_compatibility("card_history_v2", None)


def test_model_columns_must_match_contract():
    columns = main.contract_feature_names() + ["lat"]
    with pytest.raises(RuntimeError, match="differ"):
        main.check_compatibility(FEATURE_SET, columns)


def test_model_column_order_is_kept():
    columns = list(reversed(main.contract_feature_names()))
    assert main.check_compatibility(FEATURE_SET, columns) == columns
