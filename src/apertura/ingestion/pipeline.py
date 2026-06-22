from pathlib import Path

from apertura.config import get_settings
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.pdf_render import render_pdf
from apertura.ingestion.vector_store import VectorStore


def ingest_pdf(
    pdf_path: str | Path,
    doc_id: str | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> dict:
    """Render -> embed -> upsert a single PDF into Qdrant.

    Page images are written to <image_out_dir>/<doc_id>/page_NNNN.jpg so the
    frontend can later display the matched page and highlight a region on it.
    """
    settings = get_settings()
    pdf_path = Path(pdf_path)
    doc_id = doc_id or pdf_path.stem

    embedder = embedder or Embedder()
    store = store or VectorStore()
    store.ensure_collection()

    images = render_pdf(pdf_path)
    out_dir = Path(settings.image_out_dir) / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = settings.embed_batch_size
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        embeddings = embedder.embed_images(chunk)
        for offset, (image, multivector) in enumerate(zip(chunk, embeddings)):
            page_num = start + offset + 1
            image_path = out_dir / f"page_{page_num:04d}.jpg"
            image.save(image_path, "JPEG", quality=90)
            store.upsert_page(
                doc_id=doc_id,
                page_num=page_num,
                multivector=multivector,
                image_path=str(image_path),
                source=str(pdf_path),
            )

    return {"doc_id": doc_id, "pages": len(images)}
