"""Apertura LangGraph pipeline.

Graph:  router → retriever → reranker → answerer → verifier
                                                        │
                              (retry if confidence < 0.6 and retry_count < 2)
                                                        ↓
                                                    [done]

Usage:
    from apertura.agents.pipeline import build_pipeline, run_pipeline
    from apertura.ingestion.embedder import Embedder
    from apertura.ingestion.vector_store import VectorStore

    embedder = Embedder()
    store = VectorStore()
    graph = build_pipeline(embedder, store)
    result = run_pipeline(graph, question="What was iPhone revenue?")
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from apertura.agents.nodes import (
    answerer_node,
    reranker_node,
    retriever_node,
    router_node,
    set_singletons,
    verifier_node,
)
from apertura.agents.state import PipelineState
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.vector_store import VectorStore

CONFIDENCE_THRESHOLD = 0.6
MAX_RETRIES = 2


def _should_retry(state: PipelineState) -> str:
    """Edge condition: retry retrieval if verifier is not confident."""
    if not state.verified and state.confidence < CONFIDENCE_THRESHOLD and state.retry_count < MAX_RETRIES:
        return "retry"
    return "done"


def _increment_retry(state: PipelineState) -> PipelineState:
    state.retry_count += 1
    return state


def build_pipeline(embedder: Embedder, store: VectorStore) -> StateGraph:
    set_singletons(embedder, store)

    graph = StateGraph(PipelineState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    graph.add_node("router",    router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker",  reranker_node)
    graph.add_node("answerer",  answerer_node)
    graph.add_node("verifier",  verifier_node)
    graph.add_node("retry",     _increment_retry)

    # ── Add edges ──────────────────────────────────────────────────────────
    graph.add_edge(START,       "router")
    graph.add_edge("router",    "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker",  "answerer")
    graph.add_edge("answerer",  "verifier")

    # Conditional: retry or finish
    graph.add_conditional_edges(
        "verifier",
        _should_retry,
        {"retry": "retry", "done": END},
    )
    graph.add_edge("retry", "retriever")   # loop back to retrieval

    return graph.compile()


def run_pipeline(graph, question: str, doc_id: str = "default") -> PipelineState:
    initial = PipelineState(question=question, doc_id=doc_id)
    final = graph.invoke(initial)
    # LangGraph returns a dict when using dataclass state
    if isinstance(final, dict):
        result = PipelineState(**final)
    else:
        result = final
    return result
