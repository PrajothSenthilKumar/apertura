"""Apertura FastAPI gateway.

Loads ColQwen2.5 locally (dev) or delegates to Modal GPU (production).
Serves:
  POST /ingest              — index a PDF
  POST /query               — LangGraph pipeline answer
  GET  /suggest-questions   — sample questions for the document
  GET  /pages/*             — static page image serving (local only)
  GET  /health              — liveness check
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apertura.agents.observability import trace_pipeline_run
from apertura.agents.pipeline import build_pipeline, run_pipeline
from apertura.config import get_settings
from apertura.ingestion.pipeline import ingest_pdf
from apertura.ingestion.vector_store import VectorStore

USE_MODAL = os.getenv("USE_MODAL", "false").lower() == "true"

_embedder = None
_store: VectorStore | None = None
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store, _graph
    _store = VectorStore()
    _store.ensure_collection()

    if not USE_MODAL:
        from apertura.ingestion.embedder import Embedder
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

from starlette.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)



# Serve page images only in local mode (Render has no persistent disk)
if not USE_MODAL:
    pages_dir = Path("data/pages")
    pages_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/pages", StaticFiles(directory=str(pages_dir)), name="pages")


# ── Models ────────────────────────────────────────────────────────────────────
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


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "modal_mode": USE_MODAL}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    doc_id = Path(file.filename).stem
    pdf_bytes = await file.read()

    if USE_MODAL:
        from apertura.ingestion.modal_client import ingest_via_modal
        result = await ingest_via_modal(pdf_bytes, doc_id)
    else:
        tmp = Path("data/uploads") / file.filename
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(pdf_bytes)
        result = ingest_pdf(tmp, doc_id=doc_id, embedder=_embedder, store=_store)

    return IngestResponse(doc_id=result["doc_id"], pages=result["pages"])


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
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


@app.get("/suggest-questions/{doc_id}")
async def suggest_questions(doc_id: str):
    """Generate document-specific sample questions.

    In Modal/cloud mode: page images aren't on this server, so we
    generate questions from a text search of Qdrant metadata instead.
    In local mode: we use the actual page images for better questions.
    """
    import base64
    from anthropic import Anthropic
    settings = get_settings()

    GENERIC = [
        "What was total revenue for the quarter and how did it change year over year?",
        "What was net income?",
        "What was the gross margin percentage?",
        "What was earnings per share?",
        "What is the revenue outlook for next quarter?",
    ]

    # In cloud mode, generate questions from text only (no images)
    if USE_MODAL:
        try:
            client = Anthropic(api_key=settings.anthropic_api_key)
            resp = client.messages.create(
                model=settings.answer_model,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Generate exactly 5 specific analyst questions for a financial document "
                        f"with doc_id '{doc_id}'. Return ONLY a JSON array of 5 strings, no other text."
                    )
                }],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
            questions = json.loads(raw)
            return {"questions": questions[:5]}
        except Exception:
            return {"questions": GENERIC}

    # Local mode: use actual page images for better questions
    if _embedder is None:
        return {"questions": GENERIC}

    try:
        query_vec = _embedder.embed_query("revenue income financial results")
        hits = _store.search(query_vec, limit=2, doc_id_filter=doc_id)
    except Exception:
        return {"questions": GENERIC}

    if not hits:
        return {"questions": GENERIC}

    content = []
    for h in hits:
        image_path = h.payload.get("image_path", "")
        try:
            with open(image_path, "rb") as f:
                data = base64.standard_b64encode(f.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data}
            })
        except (FileNotFoundError, OSError):
            continue

    if not content:
        return {"questions": GENERIC}

    content.append({
        "type": "text",
        "text": (
            "Based on these document pages, generate exactly 5 specific useful questions "
            "an analyst would ask. Return ONLY a JSON array of 5 strings, no other text."
        )
    })

    try:
        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.answer_model,
            max_tokens=300,
            messages=[{"role": "user", "content": content}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        questions = json.loads(raw)
        return {"questions": questions[:5]}
    except Exception:
        return {"questions": GENERIC}