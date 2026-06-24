"""Shared state object that flows through the LangGraph pipeline."""

from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field


@dataclass
class PipelineState:
    # ── Input ──────────────────────────────────────────────────────────────
    question: str = ""
    doc_id: str = "default"

    # ── Router ─────────────────────────────────────────────────────────────
    query_type: str = ""          # "visual" | "text"

    # ── Retriever ──────────────────────────────────────────────────────────
    retrieved_pages: list[dict] = field(default_factory=list)
    # each dict: {page_num, image_path, score, doc_id}

    # ── Reranker ───────────────────────────────────────────────────────────
    reranked_pages: list[dict] = field(default_factory=list)

    # ── Answerer ───────────────────────────────────────────────────────────
    answer: str = ""

    # ── Verifier ───────────────────────────────────────────────────────────
    confidence: float = 0.0       # 0-1
    verified: bool = False
    retry_count: int = 0

    # ── Observability ──────────────────────────────────────────────────────
    trace_id: str = ""
    token_usage: dict[str, Any] = field(default_factory=dict)
    latencies: dict[str, float] = field(default_factory=dict)
