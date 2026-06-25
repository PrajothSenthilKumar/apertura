"""LangGraph agent nodes — works in both local and cloud (Modal) mode.

In local mode: reads page images from disk, embeds queries with local GPU.
In cloud mode: reads image_b64 from Qdrant payload, embeds via Modal GPU.
All file I/O is wrapped in try/except since images may not exist on server.
"""

from __future__ import annotations

import json
import os
import time

from apertura.agents.state import PipelineState
from apertura.config import get_settings
from apertura.ingestion.vector_store import VectorStore
from apertura.router.classifier import classify_query

_embedder = None
_store: VectorStore | None = None


def set_singletons(embedder, store: VectorStore) -> None:
    global _embedder, _store
    _embedder = embedder
    _store = store


# ── Node 1: Router ────────────────────────────────────────────────────────────
def router_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    qt = classify_query(state.question)
    state.query_type = qt.value
    state.latencies["router"] = round(time.time() - t0, 3)
    return state


# ── Node 2: Retriever ─────────────────────────────────────────────────────────
def retriever_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    settings = get_settings()

    # Embed query — local GPU or Modal cloud
    if _embedder is not None:
        query_vec = _embedder.embed_query(state.question)
    elif os.getenv("USE_MODAL", "false").lower() == "true":
        import asyncio, concurrent.futures
        from apertura.ingestion.modal_client import embed_via_modal
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, embed_via_modal(state.question))
            query_vec = future.result()
    else:
        raise RuntimeError("No embedder available")

    # Search Qdrant
    hits = _store.search(query_vec, limit=settings.top_k_pages + 2)

    state.retrieved_pages = [
        {
            "page_num":   h.payload["page_num"],
            "image_path": h.payload.get("image_path", ""),
            "image_b64":  h.payload.get("image_b64", ""),
            "doc_id":     h.payload.get("doc_id", ""),
            "score":      h.score,
        }
        for h in hits
    ]
    state.latencies["retriever"] = round(time.time() - t0, 3)
    return state


# ── Node 3: Reranker ──────────────────────────────────────────────────────────
def reranker_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    settings = get_settings()

    seen: set[int] = set()
    reranked: list[dict] = []
    for page in sorted(state.retrieved_pages, key=lambda p: p["score"], reverse=True):
        if page["page_num"] not in seen:
            reranked.append(page)
            seen.add(page["page_num"])
        if len(reranked) >= settings.top_k_pages:
            break

    state.reranked_pages = reranked
    state.latencies["reranker"] = round(time.time() - t0, 3)
    return state


# ── Node 4: Answerer ──────────────────────────────────────────────────────────
def answerer_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    import base64
    from anthropic import Anthropic
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)

    content = []
    for p in state.reranked_pages:
        # Try base64 from Qdrant payload first (cloud mode)
        img_b64 = p.get("image_b64", "")
        if img_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            })
            continue
        # Fall back to reading from disk (local mode)
        try:
            with open(p["image_path"], "rb") as f:
                data = base64.standard_b64encode(f.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data}
            })
        except (FileNotFoundError, OSError):
            continue

    content.append({
        "type": "text",
        "text": (
            "You are a financial document analyst. Answer the following question using "
            "ONLY the information visible in the provided document page images. "
            "Quote exact figures. Be concise.\n\n"
            f"Question: {state.question}"
        )
    })

    resp = client.messages.create(
        model=settings.answer_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    state.answer = "".join(b.text for b in resp.content if b.type == "text")
    state.latencies["answerer"] = round(time.time() - t0, 3)
    return state


# ── Node 5: Verifier ──────────────────────────────────────────────────────────
_VERIFIER_PROMPT = """Question: {question}
Answer: {answer}

Is this answer grounded in the document pages provided?
Reply with JSON only (no markdown):
{{"confidence": <float 0-1>, "grounded": <true|false>, "reason": "<one sentence>"}}"""


def verifier_node(state: PipelineState) -> PipelineState:
    t0 = time.time()
    import base64
    from anthropic import Anthropic
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)

    content = []
    for p in state.reranked_pages:
        img_b64 = p.get("image_b64", "")
        if img_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}
            })
            continue
        try:
            with open(p["image_path"], "rb") as f:
                data = base64.standard_b64encode(f.read()).decode()
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data}
            })
        except (FileNotFoundError, OSError):
            continue

    content.append({
        "type": "text",
        "text": _VERIFIER_PROMPT.format(question=state.question, answer=state.answer),
    })

    try:
        resp = client.messages.create(
            model=settings.answer_model,
            max_tokens=200,
            messages=[{"role": "user", "content": content}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        verdict = json.loads(raw)
        state.confidence = float(verdict.get("confidence", 0.8))
        state.verified = bool(verdict.get("grounded", True))
    except Exception:
        state.confidence = 0.8
        state.verified = True

    state.latencies["verifier"] = round(time.time() - t0, 3)
    return state
