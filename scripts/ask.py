import argparse

from apertura.answer.answerer import answer_question
from apertura.config import get_settings
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about ingested docs.")
    parser.add_argument("question")
    args = parser.parse_args()

    settings = get_settings()
    embedder = Embedder()
    store = VectorStore()

    query_vec = embedder.embed_query(args.question)
    hits = store.search(query_vec, limit=settings.top_k_pages)

    if not hits:
        print("No pages found. Ingest a document first.")
        return

    page_paths = [h.payload["image_path"] for h in hits]
    citations = ", ".join(
        f"p.{h.payload['page_num']}" for h in hits
    )

    print("Reading pages:", citations)
    answer = answer_question(args.question, page_paths)
    print("\n" + answer)
    print(f"\n[Sources: {citations}]")


if __name__ == "__main__":
    main()