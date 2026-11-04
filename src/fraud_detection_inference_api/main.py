"""Fraud detection inference API.

At startup, the API loads the model that carries the production alias
in the MLflow registry (full pipeline: preprocessing + Random Forest),
with the decision threshold of its training run, and checks that the
model matches the input contract it serves:
- the run's "feature_set" parameter equals the contract FEATURE_SET;
- the columns the pipeline was fitted on equal the contract fields.
If either check fails, the API refuses to start: a mismatch between
model and contract is detected at deployment, not at the first request.

/predict is protected by an API key (header X-API-Key).
"""

import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from mlflow import MlflowClient

from fraud_detection_contracts import (
    FEATURE_SET,
    FraudDetectionInferenceInput,
    FraudDetectionInferenceOutput,
)

# Local development reads secrets.env; on the Hugging Face Space, the
# same variables are defined as Space secrets.
load_dotenv("secrets.env")

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return value


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    model_name: str
    model_version: str
    run_id: str
    threshold: float
    feature_set: str
    feature_names: list[str]


# ---------------------------------------------------------------------
# Model loading and compatibility checks
# ---------------------------------------------------------------------

def contract_feature_names() -> list[str]:
    """Feature columns defined by the input contract."""
    transaction = FraudDetectionInferenceInput.model_fields["transaction"].annotation
    card_features = FraudDetectionInferenceInput.model_fields["card_features"].annotation
    return list(transaction.model_fields) + list(card_features.model_fields)


def check_compatibility(
    run_feature_set: str | None,
    model_feature_names: list[str] | None,
) -> list[str]:
    """Check that the model matches the contract; return its feature order."""
    if run_feature_set != FEATURE_SET:
        raise RuntimeError(
            f"Model feature set {run_feature_set!r} does not match the "
            f"contract feature set {FEATURE_SET!r}. Deploy the contract "
            "version of this model, or move the alias to a compatible model."
        )

    expected = contract_feature_names()

    if model_feature_names is None:
        return expected

    if set(model_feature_names) != set(expected):
        missing = sorted(set(model_feature_names) - set(expected))
        unexpected = sorted(set(expected) - set(model_feature_names))
        raise RuntimeError(
            "Model columns and contract fields differ. "
            f"Expected by the model, absent from the contract: {missing}. "
            f"In the contract, unknown to the model: {unexpected}."
        )

    # Keep the column order the model was fitted with
    return list(model_feature_names)


def load_production_model() -> LoadedModel:
    """Load the model carrying the alias, with its run parameters."""
    mlflow.set_tracking_uri(required_env("MLFLOW_TRACKING_URI"))
    model_name = required_env("MODEL_NAME")
    model_alias = required_env("MODEL_ALIAS")

    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, model_alias)
    run = client.get_run(model_version.run_id)
    params = run.data.params

    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version.version}")

    feature_names = check_compatibility(
        run_feature_set=params.get("feature_set"),
        model_feature_names=(
            # Since scikit-learn 1.0 fit(X,y), registers column names in
            # feature_names_in_ when X is of type pandas Dataframe.
            # _ suffix is a scikit-learn naming convention for attributes learned during fit
            # like classes_ or n_features_in_
            #  Since here we have logged pipleine with preprocessing step, attribute is taken from first step
            # hence ColumnTransformer. Attrribute contains raw columns pipeline input
            
            list(model.feature_names_in_)
            if hasattr(model, "feature_names_in_")
            else None
        ),
    )

    loaded = LoadedModel(
        model=model,
        model_name=model_name,
        model_version=str(model_version.version),
        run_id=str(model_version.run_id),
        threshold=float(params["threshold"]),
        feature_set=params["feature_set"],
        feature_names=feature_names,
    )

    print(
        f"Model loaded: {loaded.model_name} version {loaded.model_version} "
        f"(alias {model_alias}, feature set {loaded.feature_set}, "
        f"threshold {loaded.threshold})"
    )
    return loaded


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail at startup rather than at the first request
    app.state.api_key = required_env("INFERENCE_API_KEY")
    app.state.loaded_model = load_production_model()
    yield


app = FastAPI(
    title="Fraud Detection Inference API",
    version="0.2.0",
    description=(
        f"Fraud prediction for one card payment. Input contract: {FEATURE_SET} "
        "(no personal data; unknown fields are rejected)."
    ),
    lifespan=lifespan,
)


def require_api_key(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> None:
    """Reject requests without the expected X-API-Key header."""
    expected = request.app.state.api_key
    # Constant-time comparison: does not leak the key through timing
    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    loaded: LoadedModel = request.app.state.loaded_model
    return {
        "status": "ok",
        "model_name": loaded.model_name,
        "model_version": loaded.model_version,
        "feature_set": loaded.feature_set,
    }


@app.post(
    "/predict",
    response_model=FraudDetectionInferenceOutput,
    dependencies=[Depends(require_api_key)],
)
def predict(
    payload: FraudDetectionInferenceInput,
    request: Request,
) -> FraudDetectionInferenceOutput:
    loaded: LoadedModel = request.app.state.loaded_model

    # One row, columns built from the contract, in the model's order
    X = pd.DataFrame([payload.to_features()], columns=loaded.feature_names)

    fraud_probability = float(loaded.model.predict_proba(X)[0, 1])

    return FraudDetectionInferenceOutput(
        fraud_probability=fraud_probability,
        fraud_prediction=fraud_probability >= loaded.threshold,
        model_name=loaded.model_name,
        model_version=loaded.model_version,
        feature_set=loaded.feature_set,
    )
