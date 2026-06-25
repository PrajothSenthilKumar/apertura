"""Apertura FastAPI gateway.

Loads ColQwen2.5 and the Qdrant client once at startup, then serves:
  POST /ingest  — index a PDF that has already been uploaded to the server
  POST /query   — full LangGraph pipeline (router→retriever→reranker→answerer→verifier)
  GET  /health  — liveness check
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from apertura.agents.observability import trace_pipeline_run
from apertura.agents.pipeline import build_pipeline, run_pipeline
from apertura.api.static import mount_static
from apertura.config import get_settings
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.pipeline import ingest_pdf
from apertura.ingestion.vector_store import VectorStore

# ── shared singletons loaded once at startup ──────────────────────────────────
_embedder: Embedder | None = None
_store: VectorStore | None = None
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store, _graph
    from apertura.ingestion.modal_client import is_modal_enabled
    _store = VectorStore()
    _store.ensure_collection()
    if not is_modal_enabled():
        print("Loading ColQwen2.5 locally …")
        _embedder = Embedder()
    else:
        print("Modal mode — skipping local GPU load …")
        _embedder = None
    print("Building LangGraph pipeline …")
    _graph = build_pipeline(_embedder, _store)
    print("Ready.")
    yield


app = FastAPI(title="Apertura", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mount_static(app)


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
    confidence: float
    verified: bool
    latencies: dict
    trace_url: str | None = None


class IngestResponse(BaseModel):
    doc_id: str
    pages: int


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/suggest-questions/{doc_id}")
def suggest_questions(doc_id: str):
    """Generate 5 relevant sample questions for the indexed document."""
    from anthropic import Anthropic
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)

    # retrieve a couple of pages to give Claude context about the document
    query_vec = _embedder.embed_query("revenue income financial results")
    hits = _store.search(query_vec, limit=2)
    if not hits:
        return {"questions": [
            "What was total revenue for the quarter?",
            "What was net income?",
            "What was the gross margin?",
            "What was earnings per share?",
            "What is the outlook for next quarter?",
        ]}

    import base64
    content = []
    for h in hits:
        with open(h.payload["image_path"], "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})

    content.append({"type": "text", "text": (
        "Based on these document pages, generate exactly 5 specific, useful questions "
        "an analyst would ask about this document. Return ONLY a JSON array of 5 strings, "
        "no other text. Example: [\"What was revenue?\", ...]"
    )})

    resp = client.messages.create(
        model=settings.answer_model,
        max_tokens=300,
        messages=[{"role": "user", "content": content}],
    )
    import json
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    try:
        questions = json.loads(raw)
        return {"questions": questions[:5]}
    except Exception:
        return {"questions": [
            "What was total revenue for the quarter?",
            "What was net income?",
            "What was the gross margin?",
            "What was earnings per share?",
            "What is the outlook for next quarter?",
        ]}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    doc_id = Path(file.filename).stem
    pdf_bytes = await file.read()

    from apertura.ingestion.modal_client import is_modal_enabled, ingest_via_modal
    if is_modal_enabled():
        result = ingest_via_modal(pdf_bytes, doc_id)
    else:
        tmp = Path("data/uploads") / file.filename
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(pdf_bytes)
        result = ingest_pdf(tmp, doc_id=doc_id, embedder=_embedder, store=_store)

    return IngestResponse(doc_id=result["doc_id"], pages=result["pages"])


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run the full LangGraph multi-agent pipeline."""
    if not _graph:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    result = run_pipeline(_graph, question=req.question, doc_id=req.doc_id)

    if not result.answer:
        raise HTTPException(status_code=404, detail="No indexed pages found. Ingest a document first.")

    trace_url = trace_pipeline_run(req.question, result, req.doc_id)

    return QueryResponse(
        answer=result.answer,
        pages=[p["page_num"] for p in result.reranked_pages],
        image_paths=[p["image_path"] for p in result.reranked_pages],
        doc_id=req.doc_id,
        query_type=result.query_type,
        confidence=result.confidence,
        verified=result.verified,
        latencies=result.latencies,
        trace_url=trace_url,
    )
