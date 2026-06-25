"""Modal client — calls GPU functions on Modal instead of local GPU."""
import os


def is_modal_enabled() -> bool:
    return os.getenv("USE_MODAL", "false").lower() == "true"


def ingest_via_modal(pdf_bytes: bytes, doc_id: str) -> dict:
    import modal
    fn = modal.Function.lookup("apertura", "ingest_document")
    return fn.remote(pdf_bytes, doc_id)


def embed_via_modal(question: str) -> list[list[float]]:
    import modal
    fn = modal.Function.lookup("apertura", "embed_query")
    return fn.remote(question)