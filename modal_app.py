import modal
import os

app = modal.App("apertura")

# Build the container image with all dependencies
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

# Store your secrets in Modal — run: modal secret create apertura-secrets
# with QDRANT_URL, QDRANT_API_KEY, COLPALI_MODEL
@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    memory=10240,
    secrets=[modal.Secret.from_name("apertura-secrets")],
    volumes={"/model-cache": modal.Volume.from_name("apertura-model-cache", create_if_missing=True)},
)
def ingest_document(pdf_bytes: bytes, doc_id: str) -> dict:
    import os, tempfile
    from pathlib import Path

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    # inline import so Modal can find them in the container
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    import torch
    from pdf2image import convert_from_path
    from qdrant_client import QdrantClient, models
    import uuid

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

    # Save page images to /tmp for URL serving later
    pages_dir = Path(f"/tmp/pages/{doc_id}")
    pages_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(pages_dir / f"page_{i+1:04d}.jpg", "JPEG", quality=90)

    # Embed pages
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

    batch_size = 1
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        with torch.no_grad():
            batch = processor.process_images(chunk).to(device)
            outputs = model(**batch)
        embeddings = [emb.float().cpu().tolist() for emb in outputs]

        for offset, (img, emb) in enumerate(zip(chunk, embeddings)):
            page_num = start + offset + 1
            image_path = str(pages_dir / f"page_{page_num:04d}.jpg")
            client.upsert(
                collection_name=collection,
                points=[models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "image_path": image_path,
                    },
                )],
            )
        print(f"Embedded pages {start+1}-{min(start+batch_size, len(images))}/{len(images)}")

    return {"doc_id": doc_id, "pages": len(images)}


@app.function(
    image=image,
    gpu="T4",
    timeout=60,
    memory=10240,
    secrets=[modal.Secret.from_name("apertura-secrets")],
    volumes={"/model-cache": modal.Volume.from_name("apertura-model-cache", create_if_missing=True)},
)
def embed_query(question: str) -> list[list[float]]:
    import os, torch
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