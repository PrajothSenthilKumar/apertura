import argparse

from apertura.ingestion.pipeline import ingest_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into Qdrant.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--doc-id", default=None, help="Override the document id")
    args = parser.parse_args()

    result = ingest_pdf(args.pdf, doc_id=args.doc_id)
    print(f"Ingested {result['pages']} pages from '{result['doc_id']}'")


if __name__ == "__main__":
    main()
