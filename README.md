# Apertura

**Multimodal financial document RAG** — answers questions from the charts and tables in your documents, not just the text.

Apertura embeds every PDF page as an image using ColQwen2.5, retrieves visually with Qdrant's multi-vector MaxSim index, and answers with Claude vision reading the actual page. A five-node LangGraph pipeline — Router → Retriever → Reranker → Answerer → Verifier — handles every query with full observability via Langfuse.

---

## Results

Evaluated on a 30-question golden set against Apple's Q1 2026 Form 10-Q:

| System | Answer Accuracy | Table Questions | Retrieval Accuracy |
|---|---|---|---|
| Apertura (visual RAG) | **96.7%** | **96.4%** | 83.3% |
| Text-RAG baseline | 76.7% | 78.6% | 53.3% |

**+17.8% lift on table and chart questions** — the content that text extraction destroys.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Next.js UI                           │
│          Upload PDF · Ask questions · View citations        │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────────────┐
│                   FastAPI Gateway                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              LangGraph Pipeline                             │
│                                                             │
│  ┌────────┐  ┌───────────┐  ┌──────────┐                   │
│  │ Router │→ │ Retriever │→ │ Reranker │                   │
│  └────────┘  └───────────┘  └──────────┘                   │
│   classify    ColQwen2.5      deduplicate                   │
│   question    + Qdrant        + reorder                     │
│                                    │                        │
│               ┌────────────────────▼────────┐              │
│               │        Answerer             │              │
│               │   Claude vision reads       │              │
│               │   retrieved page images     │              │
│               └────────────────────┬────────┘              │
│                                    │                        │
│               ┌────────────────────▼────────┐              │
│               │        Verifier             │              │
│               │  confidence check — retry   │              │
│               │  if confidence < 0.6        │              │
│               └─────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
┌──────▼──────┐ ┌─────▼─────┐ ┌────▼────┐
│   Qdrant    │ │    S3 /   │ │Langfuse │
│multi-vector │ │local pages│ │ traces  │
└─────────────┘ └───────────┘ └─────────┘
```

**Why visual retrieval beats text extraction:** ColQwen2.5 embeds the rendered page image directly, preserving the layout, tables, and charts that OCR destroys. Qdrant stores one multi-vector representation per page and scores queries with late-interaction MaxSim — the same algorithm as ColBERT.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Visual retrieval model | ColQwen2.5 (`vidore/colqwen2.5-v0.2`) |
| Vector database | Qdrant (multi-vector, MaxSim) |
| Agent orchestration | LangGraph 5-node pipeline |
| Vision answerer | Claude (`claude-sonnet-4-6`) |
| Query router | Claude Haiku (zero-shot classification) |
| API | FastAPI |
| PDF rendering | pdf2image + poppler |
| Frontend | Next.js + Tailwind CSS |
| Observability | Langfuse |
| Eval framework | Custom golden set + text-RAG baseline |
| Infra (next) | Docker · Kubernetes · AWS Lambda |

---

## Setup

**Prerequisites:** Python 3.12, Docker Desktop, poppler, Node.js 18+

```bash
# 1. Clone and install
git clone https://github.com/PrajothSenthilKumar/apertura
cd apertura
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install python-multipart langfuse langgraph

# 2. Configure
cp .env.example .env
# Add: ANTHROPIC_API_KEY=sk-ant-...

# 3. Start Qdrant
docker compose up -d

# 4. Start backend (loads ColQwen2.5 once, stays in memory)
python scripts/serve.py

# 5. Start frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:3000`, upload a PDF, ask questions.

---

## Ingesting documents

```bash
python scripts/ingest.py path/to/filing.pdf --doc-id apple-10q
```

Good sources: SEC EDGAR 10-Qs and annual reports, company investor-relations earnings decks.

---

## Running the eval

```bash
# Full comparison: visual RAG vs text-RAG baseline (30 questions, ~15 min)
python scripts/run_eval.py --pdf 10QQ12026.pdf --doc-id apple-10q

# Fast CI gate: 10 questions, fails if accuracy drops below 80%
python scripts/run_ci_eval.py --pdf 10QQ12026.pdf
```

---

## Observability (Langfuse)

Add to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Every pipeline run is traced with per-agent latency, token cost, confidence score, and a link back to the source pages. Visit `https://cloud.langfuse.com` to view dashboards.

---

## Project structure

```
apertura/
├── src/apertura/
│   ├── agents/          LangGraph pipeline (state, nodes, pipeline, observability)
│   ├── api/             FastAPI gateway + static file serving
│   ├── answer/          Claude vision answerer
│   ├── eval/            Eval harness + text-RAG baseline
│   ├── ingestion/       PDF render → ColQwen2.5 → Qdrant
│   └── router/          Query classifier (visual vs text)
├── frontend/            Next.js UI
├── scripts/             CLI tools (ingest, serve, ask, eval)
├── eval/                Golden set YAML + results JSON
└── docker-compose.yml   Qdrant
```
