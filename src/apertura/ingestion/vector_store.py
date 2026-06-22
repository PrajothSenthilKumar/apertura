import uuid

from qdrant_client import QdrantClient, models

from apertura.config import get_settings


class VectorStore:
    """Thin wrapper over a Qdrant multi-vector collection configured for
    ColBERT-style late interaction (MaxSim)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        self.collection = settings.collection_name
        self.dim = settings.vector_dim

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    def upsert_page(
        self,
        *,
        doc_id: str,
        page_num: int,
        multivector: list[list[float]],
        image_path: str,
        source: str | None = None,
    ) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=multivector,
                    payload={
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "image_path": image_path,
                        "source": source,
                    },
                )
            ],
        )

    def search(self, query_multivector: list[list[float]], limit: int = 5):
        return self.client.query_points(
            collection_name=self.collection,
            query=query_multivector,
            limit=limit,
            with_payload=True,
        ).points
