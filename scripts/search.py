import argparse

from apertura.ingestion.embedder import Embedder
from apertura.ingestion.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Search ingested pages.")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--k", type=int, default=5, help="Number of pages to return")
    args = parser.parse_args()

    embedder = Embedder()
    store = VectorStore()

    query_vec = embedder.embed_query(args.query)
    hits = store.search(query_vec, limit=args.k)

    if not hits:
        print("No results. Have you ingested a document yet?")
        return

    for hit in hits:
        payload = hit.payload
        print(
            f"{hit.score:7.3f}  {payload['doc_id']} "
            f"p.{payload['page_num']}  ->  {payload['image_path']}"
        )


if __name__ == "__main__":
    main()
