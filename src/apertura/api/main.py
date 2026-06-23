"""Apertura FastAPI gateway.

Loads ColQwen2.5 and the Qdrant client once at startup, then serves:
  POST /ingest  — index a PDF that has already been uploaded to the server
  POST /query   — retrieve pages and answer with Claude vision
  GET  /health  — liveness check
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from apertura.answer.answerer import answer_question
from apertura.config import get_settings
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.pipeline import ingest_pdf
from apertura.ingestion.vector_store import VectorStore
from apertura.router.classifier import QueryType, classify_query

# ── shared singletons loaded once at startup ──────────────────────────────────
_embedder: Embedder | None = None
_store: VectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store
    print("Loading ColQwen2.5 …")
    _embedder = Embedder()
    _store = VectorStore()
    _store.ensure_collection()
    print("Ready.")
    yield


app = FastAPI(title="Apertura", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    doc_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    pages: list[int]
    image_paths: list[str]
    doc_id: str
    query_type: str


class IngestResponse(BaseModel):
    doc_id: str
    pages: int


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload and index a PDF. doc_id is derived from the filename."""
    settings = get_settings()
    doc_id = Path(file.filename).stem
    tmp = Path("data/uploads") / file.filename
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())

    result = ingest_pdf(tmp, doc_id=doc_id, embedder=_embedder, store=_store)
    return IngestResponse(doc_id=result["doc_id"], pages=result["pages"])


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Retrieve relevant pages then ask Claude to answer from the images."""
    settings = get_settings()
    query_type = classify_query(req.question)
    query_vec = _embedder.embed_query(req.question)
    hits = _store.search(query_vec, limit=settings.top_k_pages)

    if not hits:
        raise HTTPException(status_code=404, detail="No indexed pages found. Ingest a document first.")

    page_paths = [h.payload["image_path"] for h in hits]
    page_nums  = [h.payload["page_num"]   for h in hits]

    answer = answer_question(req.question, page_paths)

    return QueryResponse(
        answer=answer,
        pages=page_nums,
        image_paths=page_paths,
        doc_id=req.doc_id,
        query_type=query_type.value,
    )
