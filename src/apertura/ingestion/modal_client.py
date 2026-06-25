"""Modal client — calls GPU functions on Modal instead of local GPU."""
import os


def is_modal_enabled() -> bool:
    return os.getenv("USE_MODAL", "false").lower() == "true"


def ingest_via_modal(pdf_bytes: bytes, doc_id: str) -> dict:
    import modal
    with modal.enable_output():
        f = modal.Function.from_name("apertura", "ingest_document")
        result = f.remote(pdf_bytes, doc_id)
    return result


def embed_via_modal(question: str) -> list[list[float]]:
    import modal
    f = modal.Function.from_name("apertura", "embed_query")
    return f.remote(question)