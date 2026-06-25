# Apertura

**Multimodal financial document RAG** — answers questions from the charts and tables in your documents, not just the text.

**Live demo:** https://apertura-rho.vercel.app &nbsp;|&nbsp; [GitHub](https://github.com/PrajothSenthilKumar/apertura)

---

## Results

Evaluated on a 30-question golden set against Apple's Q1 2026 Form 10-Q:

| System | Answer Accuracy | Table Questions | Retrieval Accuracy |
|---|---|---|---|
| Apertura (visual RAG) | **96.7%** | **96.4%** | 83.3% |
| Text-RAG baseline | 76.7% | 78.6% | 53.3% |

**+17.8% lift on table and chart questions** — the content that text extraction destroys.

---

## What it does

Upload any financial filing (10-K, 10-Q, 8-K, earnings deck). Apertura embeds every page as an image using ColQwen2.5, retrieves the most relevant pages visually, and answers your question with Claude reading the actual table or chart — not mangled OCR text.

Every answer comes with a confidence score, the source page citations, and actual page thumbnails showing exactly where the answer came from.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Next.js UI (Vercel)                       │
│         Upload PDF · Ask questions · View citations         │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTPS
┌─────────────────────▼───────────────────────────────────────┐
│               FastAPI Gateway (Render)                      │
│          Routes /ingest → Modal, /query → LangGraph         │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
┌──────────▼──────────┐   ┌───────────▼───────────────────────┐
│   Modal GPU (T4)    │   │     LangGraph Pipeline             │
│                     │   │                                    │
│  PDF → page images  │   │  ┌────────┐   ┌───────────┐       │
│  ColQwen2.5 embed   │   │  │ Router │ → │ Retriever │       │
│  Write to Qdrant    │   │  └────────┘   └─────┬─────┘       │
└─────────┬───────────┘   │                     │             │
          │               │          ┌──────────▼──────────┐  │
          │               │          │      Reranker        │  │
          │               │          └──────────┬──────────┘  │
          │               │                     │             │
          │               │          ┌──────────▼──────────┐  │
          │               │          │  Answerer (Claude)   │  │
          │               │          │  reads page images   │  │
          │               │          └──────────┬──────────┘  │
          │               │                     │             │
          │               │          ┌──────────▼──────────┐  │
          │               │          │  Verifier            │  │
          │               │          │  confidence < 0.6    │  │
          │               │          │  → retry retrieval   │  │
          │               │          └─────────────────────┘  │
          │               └───────────────────────────────────┘
          │
┌─────────▼───────────┐   ┌───────────────────────────────────┐
│   Qdrant Cloud      │   │         Langfuse                  │
│  multi-vector index │   │  traces · cost · latency per node │
│  MaxSim scoring     │   └───────────────────────────────────┘
└─────────────────────┘
```

**Why visual retrieval beats text extraction:** ColQwen2.5 embeds the rendered page image directly, preserving tables, charts, and layout that OCR destroys. Qdrant stores a multi-vector representation per page and scores with late-interaction MaxSim — the same algorithm as ColBERT. On dense financial tables, this retrieves the right page 83% of the time versus 53% for text-extraction baseline.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Visual retrieval model | ColQwen2.5 (`vidore/colqwen2.5-v0.2`) |
| Vector database | Qdrant Cloud (multi-vector, MaxSim) |
| Agent orchestration | LangGraph — 5-node pipeline |
| Vision answerer | Claude (`claude-sonnet-4-6`) |
| Query router | Claude Haiku (zero-shot classification) |
| GPU inference | Modal.com (serverless T4, pay-per-call) |
| API gateway | FastAPI (Render.com free tier) |
| PDF rendering | pdf2image + poppler |
| Frontend | Next.js + Tailwind CSS (Vercel) |
| Observability | Langfuse traces + cost tracking |
| Eval framework | 30-question golden set + text-RAG baseline |
| Containerization | Docker (3 microservices) |
| Orchestration | Kubernetes manifests (local kind) |
| Async ingestion | AWS Lambda architecture |

---

## Deployment Architecture

```
User browser
    │
    ├── Frontend ──────────────── Vercel (free)
    │
    ├── API gateway ────────────── Render.com (free tier)
    │       │
    │       ├── /ingest ─────────── Modal.com GPU (T4, pay-per-call)
    │       │                           ColQwen2.5 embeds pages
    │       │                           Writes to Qdrant Cloud
    │       │
    │       └── /query ──────────── LangGraph pipeline (on Render)
    │                                   Calls Qdrant Cloud for retrieval
    │                                   Calls Anthropic API for answers
    │
    └── Vector store ───────────── Qdrant Cloud (free 1GB tier)
```

**Cost at portfolio scale (~50 queries/month):** ~$0-5/month total.

---

## Local Setup

**Prerequisites:** Python 3.12, Docker Desktop, poppler, Node.js 18+

```bash
# 1. Clone
git clone https://github.com/PrajothSenthilKumar/apertura
cd apertura

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install PyTorch with CUDA (Windows + NVIDIA GPU)
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Install project
pip install -e .
pip install python-multipart langfuse langgraph anthropic

# 5. Configure
cp .env.example .env
# Add: ANTHROPIC_API_KEY=sk-ant-...
#      QDRANT_URL=http://localhost:6333 (local) or Qdrant Cloud URL

# 6. Start Qdrant locally
docker compose up -d

# 7. Start backend (loads ColQwen2.5 once, keeps in memory)
python scripts/serve.py

# 8. Start frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:3000`.

---

## Usage

**Ingest a document:**
```bash
python scripts/ingest.py path/to/filing.pdf --doc-id apple-10q
```

Or upload directly through the UI — drag and drop any PDF.

**Good document sources:**
- SEC EDGAR: `https://www.sec.gov/cgi-bin/browse-edgar`
- Company IR pages: Apple, NVIDIA, Microsoft, Tesla quarterly filings
- Any earnings press release or investor presentation PDF

**Ask questions:**
```bash
python scripts/ask.py "What was total revenue for the quarter?"
```

---

## Evaluation

**Run the full eval** — visual RAG vs text-RAG baseline (30 questions, ~15 min):
```bash
python scripts/run_eval.py --pdf 10QQ12026.pdf --doc-id apple-10q
```

**Run the CI gate** — 10 questions, fails if accuracy drops below 80%:
```bash
python scripts/run_ci_eval.py --pdf 10QQ12026.pdf
```

**Results from the last run:**
```
Apertura Visual RAG
  Retrieval accuracy  : 83.3%
  Answer hit rate     : 96.7%
  Table questions     : 96.4%
  Text questions      : 100.0%
  Avg latency         : 4.15s

Text-RAG Baseline
  Retrieval accuracy  : 53.3%
  Answer hit rate     : 76.7%
  Table questions     : 78.6%
  Text questions      : 50.0%
  Avg latency         : 2.28s

★ Table-question lift: Apertura beats text RAG by 17.8%
```

---

## Observability (Langfuse)

Add to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Every pipeline run is traced with per-agent latency, token cost, confidence score, and a link back to the source pages. The UI shows a "View trace in Langfuse →" link on every answer.

---

## Production Deployment

**Docker Compose (full local stack):**
```bash
docker compose -f docker-compose.full.yml up --build
```

Runs all three microservices (ingestion, query, gateway) plus Qdrant and Grafana in Docker.

**Kubernetes (local kind cluster):**
```bash
kind create cluster --config k8s/kind-config.yaml
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml -n apertura
kubectl apply -f k8s/ -n apertura
kubectl get pods -n apertura -w
```

**Lambda async ingestion:** See `lambda/ARCHITECTURE.md` for the async S3-triggered ingestion architecture that decouples PDF processing from the query path.

---

## Project Structure

```
apertura/
├── src/apertura/
│   ├── agents/          LangGraph pipeline (state, nodes, pipeline, observability)
│   │   ├── state.py     Shared state dataclass flowing through all nodes
│   │   ├── nodes.py     Router, Retriever, Reranker, Answerer, Verifier
│   │   ├── pipeline.py  LangGraph graph definition with retry edge
│   │   └── observability.py  Langfuse tracing wrapper
│   ├── api/             FastAPI apps
│   │   ├── main.py      Single-process dev server
│   │   ├── gateway_app.py    Microservice gateway (proxy)
│   │   ├── ingestion_app.py  Ingestion microservice
│   │   ├── query_app.py      Query microservice
│   │   └── static.py    Page image static file serving
│   ├── answer/          Claude vision answerer
│   ├── eval/            Eval harness + text-RAG baseline
│   ├── ingestion/       PDF render → ColQwen2.5 → Qdrant
│   └── router/          Query classifier (visual vs text)
├── frontend/            Next.js UI (Vercel)
├── scripts/             CLI tools
│   ├── serve.py         Start dev backend
│   ├── ingest.py        Index a PDF
│   ├── ask.py           Ask a question via CLI
│   ├── run_eval.py      Full 30-question eval
│   └── run_ci_eval.py   Fast CI gate (10 questions)
├── eval/
│   ├── golden_set.yaml  30 labeled questions with expected answers
│   └── results.json     Last eval run results
├── docker/              Dockerfiles for each microservice
├── docker-compose.yml   Qdrant (local dev)
├── docker-compose.full.yml  Full stack with all microservices
├── k8s/                 Kubernetes deployment manifests
├── lambda/              AWS Lambda async ingestion function
├── grafana/             Grafana dashboard provisioning
├── blog/                Technical blog post (publish on Medium/LinkedIn)
└── modal_app.py         Modal GPU functions for cloud deployment
```

---

## Blog Post

Read the full technical writeup: [`blog/post.md`](blog/post.md)

Covers the visual retrieval architecture, the baseline study methodology, key engineering decisions, and lessons learned — particularly the transformers version compatibility issue that caused non-deterministic retrieval with randomly initialized LoRA adapters.

---

## License

MIT
