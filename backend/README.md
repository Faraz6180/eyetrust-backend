---
title: EyeTrust API
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# EyeTrust AI — Backend

FastAPI backend for AI-powered eye disease detection (MobileNetV2) with AI doctor chat (Groq).

## Endpoints

- `GET /` — root/status
- `GET /health` — health check
- `POST /api/predict` — upload an eye image, get disease prediction + AI advice
- `POST /api/generate-report` — generate a PDF report from a prediction
- `POST /api/chat` — chat with the AI doctor

## Setup (local)

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

- `GROQ_API_KEY` — required for the AI doctor chat/advice features.

## Deployment

This repo includes a `Dockerfile` configured for Hugging Face Spaces (Docker SDK, port 7860).
Set `GROQ_API_KEY` as a secret in your Space settings.

For Render, set the start command to:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```
