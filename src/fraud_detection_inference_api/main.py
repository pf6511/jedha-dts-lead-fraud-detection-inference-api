import os
from contextlib import asynccontextmanager

import mlflow
from dotenv import load_dotenv
from fastapi import FastAPI
from mlflow import MlflowClient

import pandas as pd
from fastapi import Request

from fraud_detection_contracts import (
    FraudDetectionInferenceInput,
    FraudDetectionInferenceOutput,
)

load_dotenv("secrets.env")

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_ALIAS = os.environ["MODEL_ALIAS"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    print(
        f"Model found: {MODEL_NAME} "
        f"version={model_version.version} "
        f"run_id={model_version.run_id}"
    )

    model_uri = f"models:/{MODEL_NAME}/{model_version.version}"

    print(f"Loading model from: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    print("Model loaded successfully.")

    run = client.get_run(model_version.run_id)

    threshold = float(run.data.params["threshold"])

    print(f"Model threshold: {threshold}")

    app.state.model = model
    app.state.model_name = MODEL_NAME
    app.state.model_version = model_version.version
    app.state.threshold = threshold

    yield


app = FastAPI(
    title="Fraud Detection Inference API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post(
    "/predict",
    response_model=FraudDetectionInferenceOutput,
)
def predict(
    payload: FraudDetectionInferenceInput,
    request: Request,
) -> FraudDetectionInferenceOutput:

    model = request.app.state.model
    threshold = request.app.state.threshold

    transaction = payload.transaction
    card_features = payload.card_features

    X = pd.DataFrame(
        [
            {
                "amt": transaction.amt,
                "lat": transaction.lat,
                "long": transaction.long,
                "city_pop": transaction.city_pop,
                "merch_lat": transaction.merch_lat,
                "merch_long": transaction.merch_long,
                "distance_km": card_features.distance_km,
                "card_transaction_count": card_features.card_transaction_count,
                "card_fraud_count": card_features.card_fraud_count,
                "card_fraud_rate": card_features.card_fraud_rate,
                "card_avg_amount": card_features.card_avg_amount,
                "merchant": transaction.merchant,
                "category": transaction.category,
                "gender": transaction.gender,
                "state": transaction.state,
                "job": transaction.job,
            }
        ]
    )


    fraud_probability = float(
        model.predict_proba(X)[0, 1]
    )

    fraud_prediction = fraud_probability >= threshold


    return FraudDetectionInferenceOutput(
        fraud_probability=fraud_probability,
        fraud_prediction=fraud_prediction,
    )