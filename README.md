---
title: Fraud Detection Inference API
emoji: 🚨
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
---

# Fraud Detection Inference API

FastAPI service for fraud detection inference.

The API loads the production model from MLflow at startup and exposes
prediction endpoints for the fraud detection processor.

## Endpoints

- `GET /health`
- `POST /predict`
- `GET /docs`