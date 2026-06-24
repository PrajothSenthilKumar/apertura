# Apertura query service
# Runs: LangGraph pipeline (Router → Retriever → Reranker → Answerer → Verifier)
# Requires NVIDIA GPU for ColQwen2.5 query embedding

FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.6.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu126

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir \
    "colpali-engine==0.3.9" \
    "transformers>=4.50.0,<4.51.0" \
    "langgraph" \
    "langfuse" \
    "anthropic"

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "apertura.api.query_app:app", "--host", "0.0.0.0", "--port", "8002"]
