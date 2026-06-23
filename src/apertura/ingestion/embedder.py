import torch
from PIL import Image

# For ColQwen2.5 (vidore/colqwen2.5-v0.2), swap these two imports for
# `from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor`
# and update the class references below.
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

from apertura.config import get_settings


def _resolve_device(preference: str) -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Loads ColQwen2 once and produces multi-vector embeddings for
    page images (ingestion) and text queries (retrieval)."""

    def __init__(self) -> None:
        settings = get_settings()
        self.device = _resolve_device(settings.device)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        self.model = ColQwen2_5.from_pretrained(
            settings.colpali_model,
            torch_dtype=dtype,
            device_map=self.device,
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(settings.colpali_model, max_num_visual_tokens=768)

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> list[list[list[float]]]:
        """Returns one multi-vector (list of 128-dim vectors) per image."""
        batch = self.processor.process_images(images).to(self.device)
        outputs = self.model(**batch)  # (batch, num_patches, dim)
        return [emb.float().cpu().tolist() for emb in outputs]

    @torch.no_grad()
    def embed_query(self, query: str) -> list[list[float]]:
        """Returns one multi-vector (list of 128-dim vectors) for the query."""
        batch = self.processor.process_queries([query]).to(self.device)
        outputs = self.model(**batch)  # (1, num_tokens, dim)
        return outputs[0].float().cpu().tolist()
