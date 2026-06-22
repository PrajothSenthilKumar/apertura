# Apertura — Multimodal Document Intelligence (Visual RAG)

Answer natural-language questions over visually complex documents (financial
filings, technical manuals, scientific papers) by retrieving over **page images**
instead of extracted text. This preserves tables, charts, and layout that
text-extraction RAG destroys, and lets the system show *which region* of a page
an answer came from.

Built around ColQwen2 (visual document retrieval), Qdrant (multi-vector / MaxSim),
a vision-language answerer, a LangGraph agent pipeline, and a RAGAS + Langfuse
eval/observability layer.

## Why visual retrieval

The standard RAG recipe — PDF to OCR text to chunks to single-vector embeddings —
loses exactly the content that matters most in real documents: numbers inside
charts, nested tables, multi-column layout. ColQwen2 embeds the rendered page
image directly as a multi-vector representation and scores queries with late
interaction (MaxSim), beating text pipelines on visually rich content. Because
the embeddings are multi-vector, the index must support multi-vector storage —
hence Qdrant rather than single-vector pgvector.

## Architecture

```
Frontend (Next.js / Vercel)
        |
FastAPI gateway
        |
LangGraph supervisor  (Router -> Retriever -> Reranker -> Answerer -> Verifier)
        |                              |
        v                              v
Qdrant index (ColQwen2 multi-vectors)  VLM answerer (Claude / GPT-4o vision)

Ingestion (offline):  Documents (SEC / arXiv) -> Lambda (render + ColQwen2 embed) -> Qdrant
Observability:        Langfuse traces + cost  |  RAGAS sampling  |  DeepEval CI gate
```

## Milestone 1 (this repo): ingestion + retrieval

Render a PDF to page images, embed each page with ColQwen2, store the
multi-vectors in Qdrant, and run a retrieval smoke test.

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- `poppler` for `pdf2image`:
  - macOS: `brew install poppler`
  - Debian/Ubuntu: `sudo apt-get install poppler-utils`
- A GPU helps a lot for embedding; CPU works for a handful of pages.

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env

# start Qdrant
docker compose up -d
```

### Ingest a document

Grab a real filing (any 10-K PDF from SEC EDGAR works), then:

```bash
python scripts/ingest.py path/to/filing.pdf --doc-id acme-10k-2024
```

### Search it

```bash
python scripts/search.py "What was Q3 free cash flow?"
```

You'll get the top matching pages with scores and the saved page-image paths —
those images are what the answerer (milestone 2) will read.

## Roadmap

- [x] **M1** Ingestion: PDF -> ColQwen2 -> Qdrant, plus retrieval smoke test
- [ ] **M2** Answerer: pass matched page images to a vision LLM; FastAPI `/query`
- [ ] **M3** Eval harness: golden set + RAGAS, plus a text-RAG baseline to
      quantify the lift on table/chart questions
- [ ] **M4** LangGraph agents: router, reranker, verifier with a confidence loop
- [ ] **M5** Frontend: Next.js upload + answer + highlighted source region
- [ ] **M6** LLMOps: Langfuse tracing + cost, DeepEval CI gate, Grafana dashboard
- [ ] **M7** Packaging: Dockerized services, K8s manifests, Lambda ingestion, S3

## Notes

- Default model `vidore/colqwen2-v1.0` runs out of the box. For the newer SOTA,
  set `COLPALI_MODEL=vidore/colqwen2.5-v0.2` and switch the imports in
  `src/apertura/ingestion/embedder.py` to `ColQwen2_5` / `ColQwen2_5_Processor`.
- At scale, mean-pool the page multi-vectors for a fast first-stage search and
  rerank the top candidates with the full vectors; add binary/int8 quantization
  to cut index size. Out of scope for M1.
