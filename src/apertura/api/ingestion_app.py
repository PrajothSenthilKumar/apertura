"""Apertura ingestion microservice.

Handles PDF upload → page render → ColQwen2.5 embed → Qdrant write.
Runs on port 8001. Requires GPU.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from apertura.ingestion.embedder import Embedder
from apertura.ingestion.pipeline import ingest_pdf
from apertura.ingestion.vector_store import VectorStore

_embedder: Embedder | None = None
_store: VectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store
    print("Ingestion service: loading ColQwen2.5 …")
    _embedder = Embedder()
    _store = VectorStore()
    _store.ensure_collection()
    print("Ingestion service: ready.")
    yield


app = FastAPI(title="Apertura Ingestion Service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class IngestResponse(BaseModel):
    doc_id: str
    pages: int


@app.get("/health")
def health():
    return {"status": "ok", "service": "ingestion"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    doc_id = Path(file.filename).stem
    tmp = Path("data/uploads") / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())
    result = ingest_pdf(tmp, doc_id=doc_id, embedder=_embedder, store=_store)
    return IngestResponse(doc_id=result["doc_id"], pages=result["pages"])
