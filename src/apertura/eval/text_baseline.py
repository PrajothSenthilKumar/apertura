from __future__ import annotations
from pathlib import Path
import fitz
import numpy as np

class TextBaseline:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._pages: list[dict] = []

    def index_pdf(self, pdf_path: str | Path, doc_id: str | None = None) -> int:
        pdf_path = Path(pdf_path)
        doc_id = doc_id or pdf_path.stem
        doc = fitz.open(str(pdf_path))
        texts = [page.get_text("text") for page in doc]
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            self._pages.append({"doc_id": doc_id, "page_num": i + 1, "text": text, "embedding": emb})
        return len(texts)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self._pages:
            raise RuntimeError("No pages indexed. Call index_pdf first.")
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = [float(np.dot(q_emb, p["embedding"])) for p in self._pages]
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self._pages[i], "score": scores[i], "embedding": None} for i in top_k]