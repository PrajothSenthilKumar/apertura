"""Apertura query microservice.

Runs the LangGraph pipeline for every incoming question.
Runs on port 8002. Requires GPU for ColQwen2.5 query embedding.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

from apertura.agents.observability import trace_pipeline_run
from apertura.agents.pipeline import build_pipeline, run_pipeline
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.vector_store import VectorStore

_embedder: Embedder | None = None
_store: VectorStore | None = None
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store, _graph
    print("Query service: loading ColQwen2.5 and building LangGraph …")
    _embedder = Embedder()
    _store = VectorStore()
    _store.ensure_collection()
    _graph = build_pipeline(_embedder, _store)
    print("Query service: ready.")
    yield


app = FastAPI(title="Apertura Query Service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    doc_id: str
    query_type: str
    confidence: float
    verified: bool
    latencies: dict
    trace_url: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "query"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not _graph:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    result = run_pipeline(_graph, question=req.question, doc_id=req.doc_id)
    if not result.answer:
        raise HTTPException(status_code=404, detail="No indexed pages found.")
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
def suggest_questions(doc_id: str):
    import base64, json
    from anthropic import Anthropic
    from apertura.config import get_settings
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)
    query_vec = _embedder.embed_query("revenue income financial results")
    hits = _store.search(query_vec, limit=2, doc_id_filter=doc_id)
    if not hits:
        return {"questions": ["What was total revenue?", "What was net income?",
                              "What was the gross margin?", "What was earnings per share?",
                              "What is the outlook for next quarter?"]}
    content = []
    for h in hits:
        with open(h.payload["image_path"], "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    content.append({"type": "text", "text": (
        "Based on these document pages, generate exactly 5 specific useful questions "
        "an analyst would ask. Return ONLY a JSON array of 5 strings, no other text."
    )})
    resp = client.messages.create(model=settings.answer_model, max_tokens=300,
                                   messages=[{"role": "user", "content": content}])
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    try:
        return {"questions": json.loads(raw)[:5]}
    except Exception:
        return {"questions": ["What was total revenue?", "What was net income?",
                              "What was the gross margin?", "What was EPS?",
                              "What is the outlook?"]}
