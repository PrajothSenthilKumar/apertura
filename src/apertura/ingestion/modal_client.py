"""Modal client — calls GPU functions on Modal instead of local GPU."""
import os


def is_modal_enabled() -> bool:
    return os.getenv("USE_MODAL", "false").lower() == "true"


async def ingest_via_modal(pdf_bytes: bytes, doc_id: str) -> dict:
    import modal
    f = modal.Function.from_name("apertura", "ingest_document")
    result = await f.remote.aio(pdf_bytes, doc_id)
    return result


async def embed_via_modal(question: str) -> list[list[float]]:
    import modal
    f = modal.Function.from_name("apertura", "embed_query")
    return await f.remote.aio(question)