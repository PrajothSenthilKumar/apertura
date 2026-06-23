from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ColQwen2 v1.0 is the well-documented default and runs out of the box.
    # To use the newer SOTA ColQwen2.5, set COLPALI_MODEL=vidore/colqwen2.5-v0.2
    # and switch the imports in embedder.py to ColQwen2_5 / ColQwen2_5_Processor.
    colpali_model: str = "vidore/colqwen2-v1.0"
    device: str = "auto"  # auto | cuda | mps | cpu

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "apertura_pages"

    # ColQwen2/ColPali project each page to 128-dim patch vectors.
    vector_dim: int = 128

    pdf_dpi: int = 150
    embed_batch_size: int = 4
    image_out_dir: str = "data/pages"

    # Windows only: full path to poppler's bin folder, e.g.
    # C:/poppler/poppler-24.08.0/Library/bin
    # Leave unset on macOS/Linux where poppler is on PATH.
    poppler_path: str | None = None
    # Answerer (Claude vision)
    anthropic_api_key: str | None = None
    answer_model: str = "claude-sonnet-4-6"
    top_k_pages: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
