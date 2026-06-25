import modal
import os

app = modal.App("apertura")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("poppler-utils", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0", "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "colpali-engine==0.3.9",
        "transformers>=4.50.0,<4.51.0",
        "pdf2image",
        "pillow",
        "qdrant-client>=1.12.0",
        "pydantic-settings",
        "python-dotenv",
        "peft>=0.14.0,<0.15.0",
        "accelerate",
    )
)

model_volume = modal.Volume.from_name("apertura-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    memory=10240,
    secrets=[modal.Secret.from_name("apertura-secrets")],
    volumes={"/model-cache": model_volume},
)
def ingest_document(pdf_bytes: bytes, doc_id: str) -> dict:
    import base64, tempfile, uuid
    from pathlib import Path

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    import torch
    from pdf2image import convert_from_path
    from qdrant_client import QdrantClient, models

    model_name = os.environ.get("COLPALI_MODEL", "vidore/colqwen2.5-v0.2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading {model_name} on {device} …")
    model = ColQwen2_5.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(model_name)

    # Render PDF pages
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / f"{doc_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        images = convert_from_path(str(pdf_path), dpi=150)

    # Connect to Qdrant Cloud
    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    collection = "apertura_pages"
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    # Delete existing pages for this doc to avoid duplicates
    try:
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    )]
                )
            )
        )
    except Exception:
        pass

    # Embed and upsert each page
    for i, img in enumerate(images):
        page_num = i + 1

        # Convert image to base64 for storage in Qdrant
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

        # Embed with ColQwen2.5
        with torch.no_grad():
            batch = processor.process_images([img]).to(device)
            outputs = model(**batch)
        emb = outputs[0].float().cpu().tolist()

        client.upsert(
            collection_name=collection,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "doc_id": doc_id,
                    "page_num": page_num,
                    "image_path": f"data/pages/{doc_id}/page_{page_num:04d}.jpg",
                    "image_b64": img_b64,
                },
            )],
        )
        print(f"  Page {page_num}/{len(images)} ingested")

    print(f"Done: {len(images)} pages for {doc_id}")
    return {"doc_id": doc_id, "pages": len(images)}


@app.function(
    image=image,
    gpu="T4",
    timeout=60,
    memory=10240,
    secrets=[modal.Secret.from_name("apertura-secrets")],
    volumes={"/model-cache": model_volume},
)
def embed_query(question: str) -> list[list[float]]:
    import torch
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    model_name = os.environ.get("COLPALI_MODEL", "vidore/colqwen2.5-v0.2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = ColQwen2_5.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(model_name)

    with torch.no_grad():
        batch = processor.process_queries([question]).to(device)
        outputs = model(**batch)

    return outputs[0].float().cpu().tolist()
