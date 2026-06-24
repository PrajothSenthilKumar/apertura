"""AWS Lambda — async PDF ingestion handler.

Architecture:
  1. User uploads PDF to S3 via presigned URL
  2. S3 PUT triggers this Lambda
  3. Lambda downloads PDF, renders pages, embeds with ColQwen2.5, writes to Qdrant
  4. Lambda updates DynamoDB with ingestion status (pending → complete)
  5. Frontend polls GET /ingest-status/{doc_id} until status = "complete"

This decouples ingestion from the query path — a 200-page filing processes
asynchronously while the user sees "indexing in background…" instead of
waiting 5-10 minutes for a response.

Environment variables required:
  QDRANT_URL          Qdrant endpoint (e.g., http://your-qdrant-host:6333)
  QDRANT_API_KEY      Optional Qdrant API key
  COLPALI_MODEL       vidore/colqwen2.5-v0.2
  STATUS_TABLE        DynamoDB table name for ingestion status
  AWS_REGION          AWS region

Note: Lambda needs enough memory (at least 8GB) and a container image
with CUDA + ColQwen2.5. Use AWS Lambda container images for this,
not the standard zip deployment.
"""

import json
import os
import tempfile
from pathlib import Path

import boto3

# These imports only work inside the Lambda container image
# which includes PyTorch + colpali-engine + pdf2image + poppler
try:
    from apertura.ingestion.embedder import Embedder
    from apertura.ingestion.pipeline import ingest_pdf
    from apertura.ingestion.vector_store import VectorStore
    APERTURA_AVAILABLE = True
except ImportError:
    APERTURA_AVAILABLE = False

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

_embedder = None
_store = None


def _get_singletons():
    """Lazy-load embedder and store (warm Lambda reuse)."""
    global _embedder, _store
    if _embedder is None:
        _embedder = Embedder()
    if _store is None:
        _store = VectorStore()
        _store.ensure_collection()
    return _embedder, _store


def _update_status(doc_id: str, status: str, pages: int = 0, error: str = "") -> None:
    table_name = os.environ.get("STATUS_TABLE")
    if not table_name:
        return
    table = dynamodb.Table(table_name)
    item = {"doc_id": doc_id, "status": status}
    if pages:
        item["pages"] = pages
    if error:
        item["error"] = error
    table.put_item(Item=item)


def handler(event: dict, context) -> dict:
    """S3 trigger handler.

    event["Records"][0]["s3"] contains bucket and key of the uploaded PDF.
    """
    for record in event.get("Records", []):
        s3_info = record.get("s3", {})
        bucket = s3_info["bucket"]["name"]
        key = s3_info["object"]["key"]
        doc_id = Path(key).stem

        print(f"Processing s3://{bucket}/{key} → doc_id={doc_id}")
        _update_status(doc_id, "processing")

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_pdf = Path(tmp_dir) / Path(key).name
                s3.download_file(bucket, key, str(local_pdf))

                embedder, store = _get_singletons()
                result = ingest_pdf(
                    local_pdf,
                    doc_id=doc_id,
                    embedder=embedder,
                    store=store,
                )

            _update_status(doc_id, "complete", pages=result["pages"])
            print(f"Ingested {result['pages']} pages from {doc_id}")

        except Exception as e:
            print(f"ERROR ingesting {doc_id}: {e}")
            _update_status(doc_id, "failed", error=str(e))
            return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    return {"statusCode": 200, "body": json.dumps({"status": "ok"})}
