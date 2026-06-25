import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

if not USE_MODAL:
    pages_dir = Path("data/pages")
    pages_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/pages", StaticFiles(directory=str(pages_dir)), name="pages")


class QueryRequest(BaseModel):
    question: str
    doc_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    pages: list[int]
    image_paths: list[str]
    image_b64s: list[str] = []
    doc_id: str
    query_type: str
    confidence: float
    verified: bool
    latencies: dict
    trace_url: str | None = None


class IngestResponse(BaseModel):
    doc_id: str
    pages: int


@app.get("/health")
def health():
    return {"status": "ok", "modal_mode": USE_MODAL}


@app.api_route("/ingest", methods=["POST", "OPTIONS"])
async def ingest(file: UploadFile = File(None)):
    if file is None:
        return JSONResponse({"ok": True})
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


@app.api_route("/query", methods=["POST", "OPTIONS"])
async def query(req: QueryRequest = None):
    if req is None:
        return JSONResponse({"ok": True})
    if not _graph:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    result = run_pipeline(_graph, question=req.question, doc_id=req.doc_id)
    if not result.answer:
        raise HTTPException(status_code=404, detail="No indexed pages found.")
    trace_url = trace_pipeline_run(req.question, result, req.doc_id)
    return QueryResponse(
        answer=result.answer,
        pages=[p["page_num"] for p in result.reranked_pages],
        image_paths=[p.get("image_path", "") for p in result.reranked_pages],
        image_b64s=[p.get("image_b64", "") for p in result.reranked_pages],
        doc_id=req.doc_id,
        query_type=result.query_type,
        confidence=result.confidence,
        verified=result.verified,
        latencies=result.latencies,
        trace_url=trace_url,
    )


@app.get("/suggest-questions/{doc_id}")
async def suggest_questions(doc_id: str):
    from anthropic import Anthropic
    settings = get_settings()

    GENERIC = [
        "What was total revenue for the quarter and how did it change year over year?",
        "What was net income?",
        "What was the gross margin percentage?",
        "What was earnings per share?",
        "What is the revenue outlook for next quarter?",
    ]

    if USE_MODAL:
        try:
            import base64
            from anthropic import Anthropic
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            settings = get_settings()

            # Fetch pages directly by doc_id — no Modal GPU call needed
            results = _store.client.scroll(
                collection_name=_store.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
                limit=2,
                with_payload=True,
                with_vectors=False,
            )
            hits = results[0]

            content = []
            for h in hits:
                img_b64 = h.payload.get("image_b64", "")
                if img_b64:
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
                    })

            if content:
                content.append({"type": "text", "text": (
                    "Based on these document pages, generate exactly 5 specific analyst questions. "
                    "Return ONLY a JSON array of 5 strings, no other text."
                )})
                client = Anthropic(api_key=settings.anthropic_api_key)
                resp = client.messages.create(
                    model=settings.answer_model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": content}],
                )
                raw = "".join(b.text for b in resp.content if b.type == "text").strip()
                return {"questions": json.loads(raw)[:5]}
        except Exception:
            pass
        return {"questions": GENERIC}

    # Local mode
    if _embedder is None:
        return {"questions": GENERIC}
    try:
        query_vec = _embedder.embed_query("revenue income financial results")
        hits = _store.search(query_vec, limit=2, doc_id_filter=doc_id)
        content = []
        for h in hits:
            import base64
            img_b64 = h.payload.get("image_b64", "")
            if img_b64:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}})
                continue
            try:
                with open(h.payload.get("image_path", ""), "rb") as f:
                    data = base64.standard_b64encode(f.read()).decode()
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
            except (FileNotFoundError, OSError):
                continue
        if not content:
            return {"questions": GENERIC}
        content.append({"type": "text", "text": (
            "Based on these document pages, generate exactly 5 specific analyst questions. "
            "Return ONLY a JSON array of 5 strings, no other text."
        )})
        client = Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(model=settings.answer_model, max_tokens=300,
                                       messages=[{"role": "user", "content": content}])
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        return {"questions": json.loads(raw)[:5]}
    except Exception:
        return {"questions": GENERIC}
