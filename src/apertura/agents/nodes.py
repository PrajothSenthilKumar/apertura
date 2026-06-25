"""LangGraph agent nodes.

Five nodes in order:
  router    → classify question as visual or text
  retriever → ColQwen2.5 + Qdrant search
  reranker  → cross-score and reorder candidates
  answerer  → Claude vision reads page images
  verifier  → confidence check, decides whether to retry
"""

from __future__ import annotations

import time

from apertura.agents.state import PipelineState
from apertura.answer.answerer import answer_question
from apertura.config import get_settings
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.vector_store import VectorStore
from apertura.router.classifier import classify_query

# ── singletons (injected at pipeline build time) ──────────────────────────
_embedder: Embedder | None = None
_store: VectorStore | None = None


def set_singletons(embedder: Embedder, store: VectorStore) -> None:
    global _embedder, _store
    _embedder = embedder
    _store = store


# ── Node 1: Router ────────────────────────────────────────────────────────
def router_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    qt = classify_query(state.question)
    state.query_type = qt.value
    state.latencies["router"] = round(time.time() - t0, 3)
    return state


# ── Node 2: Retriever ─────────────────────────────────────────────────────
def retriever_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    settings = get_settings()

    if _embedder is not None:
        query_vec = _embedder.embed_query(state.question)
    else:
        import os, asyncio
        if os.getenv("USE_MODAL", "false").lower() == "true":
            from apertura.ingestion.modal_client import embed_via_modal
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, embed_via_modal(state.question))
                        query_vec = future.result()
                else:
                    query_vec = loop.run_until_complete(embed_via_modal(state.question))
            except Exception as e:
                raise RuntimeError(f"Modal embed failed: {e}")
        else:
            raise RuntimeError("No embedder available")

    results = _store.search(
        query_vectors=query_vec,
        doc_id=state.doc_id,
        top_k=settings.top_k_pages,
    )
    state.retrieved_pages = results          # list of dicts with score, page_num, image_path
    state.latencies["retriever"] = round(time.time() - t0, 3)
    return state


# ── Node 3: Reranker ──────────────────────────────────────────────────────
def reranker_node(state: PipelineState) -> PipelineState:
    """Score pages by query–page relevance using the embedder's MaxSim score.

    For now we re-use the ColQwen2.5 retrieval score (already a MaxSim over
    patch vectors) but apply a page-diversity bonus so the same page doesn't
    fill all top-k slots.
    """
    t0 = time.time()
    settings = get_settings()

    seen_pages: set[int] = set()
    reranked: list[dict] = []

    # Sort by score descending, deduplicate page numbers
    sorted_pages = sorted(state.retrieved_pages, key=lambda p: p["score"], reverse=True)
    for page in sorted_pages:
        if page["page_num"] not in seen_pages:
            reranked.append(page)
            seen_pages.add(page["page_num"])
        if len(reranked) >= settings.top_k_pages:
            break

    state.reranked_pages = reranked
    state.latencies["reranker"] = round(time.time() - t0, 3)
    return state


# ── Node 4: Answerer ──────────────────────────────────────────────────────
def answerer_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    page_paths = [p["image_path"] for p in state.reranked_pages]
    state.answer = answer_question(state.question, page_paths)
    state.latencies["answerer"] = round(time.time() - t0, 3)
    return state


# ── Node 5: Verifier ──────────────────────────────────────────────────────
_VERIFIER_PROMPT = """You are checking whether an answer is grounded in the
document pages provided.

Question: {question}
Answer: {answer}

Is the answer supported by the pages? Reply with a JSON object:
{{"confidence": <float 0-1>, "grounded": <true|false>, "reason": "<one sentence>"}}

Reply with JSON only, no markdown."""

def verifier_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    from anthropic import Anthropic
    import json
    import base64

    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)

    # build content: images + verification prompt
    content = []
    for p in state.reranked_pages:
        with open(p["image_path"], "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})

    content.append({
        "type": "text",
        "text": _VERIFIER_PROMPT.format(question=state.question, answer=state.answer),
    })

    resp = client.messages.create(
        model=settings.answer_model,
        max_tokens=200,
        messages=[{"role": "user", "content": content}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()

    try:
        verdict = json.loads(raw)
        state.confidence = float(verdict.get("confidence", 0.5))
        state.verified = bool(verdict.get("grounded", False))
    except Exception:
        state.confidence = 0.5
        state.verified = True   # don't block on parse failure

    state.latencies["verifier"] = round(time.time() - t0, 3)
    return state
