# Apertura ingestion service
# Runs: PDF render + ColQwen2.5 embed + Qdrant write
# Requires NVIDIA GPU (--gpus all)

FROM python:3.12-slim

# System deps: poppler for pdf2image
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA first so pip install -e . doesn't clobber it
RUN pip install --no-cache-dir torch==2.6.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# Copy and install the project
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir "colpali-engine==0.3.9" "transformers>=4.50.0,<4.51.0"

# Pre-download the model so container startup is fast
RUN python -c "from apertura.ingestion.embedder import Embedder; Embedder()"

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "apertura.api.ingestion_app:app", "--host", "0.0.0.0", "--port", "8001"]
