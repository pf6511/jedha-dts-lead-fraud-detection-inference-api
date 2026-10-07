FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY fraud-detection-contracts ./fraud-detection-contracts
COPY src ./src

RUN uv sync --frozen --no-dev

EXPOSE 7860

CMD ["uv", "run", "python", "-m", "uvicorn", "fraud_detection_inference_api.main:app", "--host", "0.0.0.0", "--port", "7860"]