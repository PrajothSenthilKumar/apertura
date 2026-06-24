# Apertura gateway service
# Thin FastAPI router — proxies /ingest to ingestion-service
# and /query to query-service. No GPU required.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx pydantic-settings python-multipart

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "apertura.api.gateway_app:app", "--host", "0.0.0.0", "--port", "8000"]
