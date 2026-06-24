"""Langfuse observability.

Wraps the pipeline run with a Langfuse trace so every agent call is
recorded with inputs, outputs, latency, and token cost.

Set these in .env to enable (leave unset to run without tracing):
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL

If keys are missing, tracing is silently skipped — the pipeline works
the same way with or without Langfuse.
"""

from __future__ import annotations

import os
from typing import Any


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def trace_pipeline_run(
    question: str,
    result: Any,          # PipelineState
    doc_id: str = "default",
) -> str | None:
    """Create a Langfuse trace for a completed pipeline run.

    Returns the trace URL if tracing is enabled, else None.
    """
    if not _langfuse_enabled():
        return None

    try:
        from langfuse import Langfuse

        lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        trace = lf.trace(
            name="apertura-query",
            input={"question": question, "doc_id": doc_id},
            output={"answer": result.answer, "confidence": result.confidence},
            metadata={
                "query_type":    result.query_type,
                "retrieved_pages": [p["page_num"] for p in result.reranked_pages],
                "verified":      result.verified,
                "retry_count":   result.retry_count,
                "latencies":     result.latencies,
            },
            tags=["apertura", result.query_type],
        )

        # Individual spans for each agent
        for node, latency in result.latencies.items():
            trace.span(
                name=node,
                metadata={"latency_s": latency},
            )

        lf.flush()
        return trace.get_trace_url()

    except Exception as e:
        print(f"[Langfuse] tracing failed (non-fatal): {e}")
        return None
