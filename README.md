# Apertura

**Multimodal financial document RAG** --- answers questions from the charts and tables in your documents, not just the text.

**Live demo:** https://apertura-rho.vercel.app &nbsp;|&nbsp; [GitHub](https://github.com/PrajothSenthilKumar/apertura)

---

## Demo

Ask Apertura about NVIDIA's Q1 FY2027 8-K:-

> **"What was GAAP vs non-GAAP net income and what explains the difference?"**

Apertura reads the multi column reconciliation table directly from the page image and answers:

> *GAAP net income was $58,321M vs Non-GAAP net income of $45,548M. The $12,773M difference is primarily driven by a $15,936M gain from equity securities excluded under Non-GAAP reporting, partially offset by a $2,890M income tax adjustment.*

Text extraction RAG cannot answer this --> --> the reconciliation table structure is destroyed by OCR. Apertura reads the rendered page image and gets it right.

---

## Results

Evaluated on a 30 question golden set against Apple's Q1 2026 Form 10-Q:

| System | Answer Accuracy | Table Questions | Retrieval Accuracy |
|---|---|---|---|
| Apertura (visual RAG) | **96.7%** | **96.4%** | 83.3% |
| Text-RAG baseline | 76.7% | 78.6% | 53.3% |

**+17.8% lift on table and chart questions** --> the content that text extraction destroys.

---

## What it does

Upload any financial filing (10-K, 10-Q, 8-K, earnings deck). Apertura embeds every page as an image using ColQwen2.5, retrieves the most relevant pages visually with Qdrant's multi vector MaxSim index, and answers with Claude reading the actual table or chart.

Every answer comes with:
- Confidence score and verification status
- Source page citations
- Actual page thumbnails showing exactly where the answer came from
- Per agent latency breakdown

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
│     Routes /ingest → Modal GPU, /query → LangGraph          │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
┌──────────▼──────────┐   ┌───────────▼───────────────────────┐
│   Modal GPU (T4)    │   │     LangGraph Pipeline             │
│                     │   │                                    │
│  PDF → page images  │   │  ┌────────┐   ┌───────────┐       │
│  ColQwen2.5 embed   │   │  │ Router │ → │ Retriever │       │
│  image_b64 stored   │   │  └────────┘   └─────┬─────┘       │
│  in Qdrant payload  │   │                     │             │
└─────────┬───────────┘   │          ┌──────────▼──────────┐  │
          │               │          │      Reranker        │  │
          ▼               │          └──────────┬──────────┘  │
┌─────────────────────┐   │                     │             │
│   Qdrant Cloud      │   │          ┌──────────▼──────────┐  │
│  multi vector index │◄──┘          │  Answerer (Claude)  │  │
│  MaxSim scoring     │              │  reads image_b64    │  │
│  image_b64 payload  │              │  from Qdrant        │  │
└─────────────────────┘              └──────────┬──────────┘  │
                                                │             │
                                     ┌──────────▼──────────┐  │
                                     │  Verifier            │  │
                                     │  confidence < 0.6   │  │
                                     │  → retry retrieval  │  │
                                     └─────────────────────┘  │
                                     └───────────────────────┘
```

**Why visual retrieval beats text extraction:** ColQwen2.5 embeds the rendered page image directly by preserving tables, charts, and layout that OCR destroys. Page images are stored as base64 in Qdrant's payload so they travel with the vectors and are always available wherever the query runs, including cloud deployments with no local disk.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Visual retrieval model | ColQwen2.5 (`vidore/colqwen2.5-v0.2`) |
| Vector database | Qdrant Cloud (multi vector, MaxSim, image_b64 payload) |
| Agent orchestration | LangGraph with 5 nodes (Agents) pipeline with retry loop |
| Vision answerer | Claude (`claude-sonnet-4-6`) |
| Query router | Claude Haiku (zero shot classification) |
| GPU inference | Modal.com (serverless T4, pay per call) |
| API gateway | FastAPI (Render.com free tier) |
| PDF rendering | pdf2image + poppler |
| Frontend | Next.js + Tailwind CSS (Vercel) |
| Observability | Langfuse traces + cost tracking |
| Eval framework | 30-question golden set + text RAG baseline |
| Containerization | Docker (3 microservices) |
| Orchestration | Kubernetes manifests (local kind) |
| Async ingestion | AWS Lambda architecture |

---

## Cloud Deployment Architecture

```
User browser
    │
    ├── Frontend ──────────────── Vercel (free)
    │
    ├── API gateway ────────────── Render.com (free tier)
    │       │
    │       ├── /ingest ─────────── Modal.com GPU (T4, serverless)
    │       │                           ColQwen2.5 embeds pages
    │       │                           Stores image_b64 in Qdrant
    │       │
    │       └── /query ──────────── LangGraph pipeline (on Render)
    │                                   Modal GPU embeds query
    │                                   Qdrant returns pages + images
    │                                   Claude vision reads image_b64
    │
    └── Vector store ───────────── Qdrant Cloud (free 1GB tier)
                                       Stores multi vectors + image_b64
```

**Cold start note:** First query after idle has a ~30s cold start (Modal GPU spinning up + ColQwen2.5 loading). Subsequent queries are ~8-10s. This is the trade off for serverless GPU.

---

## Local Setup

**Prerequisites:** Python 3.12, Docker Desktop, poppler, Node.js 18+, NVIDIA GPU recommended

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
pip install python-multipart langfuse langgraph anthropic "colpali-engine==0.3.9" "transformers>=4.50.0,<4.51.0"

# 5. Configure
cp .env.example .env
# Add: ANTHROPIC_API_KEY=sk-ant-...

# 6. Start Qdrant
docker compose up -d

# 7. Start backend
python scripts/serve.py

# 8. Start frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:3000`

---

## Usage

**Ingest a document via CLI:**
```bash
python scripts/ingest.py path/to/filing.pdf --doc-id apple-10q
```

**Or upload through the UI** --> drag and drop any PDF.

**Good document sources:**
- SEC EDGAR: `https://www.sec.gov/cgi-bin/browse-edgar`
- Apple IR: `https://investor.apple.com`
- NVIDIA IR: `https://investor.nvidia.com`
- Microsoft IR: `https://www.microsoft.com/en-us/investor`

**Ask questions via CLI:**
```bash
python scripts/ask.py "What was total net sales for the quarter?"
```

---

## Evaluation

**Full eval --> visual RAG vs text RAG baseline (30 questions, ~15 min):**
```bash
python scripts/run_eval.py --pdf 10QQ12026.pdf --doc-id apple-10q
```

**CI gate --> 10 questions, fails if accuracy drops below 80%:**
```bash
python scripts/run_ci_eval.py --pdf 10QQ12026.pdf
```

**Last eval results:**
```
Apertura Visual RAG
  Retrieval accuracy  : 83.3%
  Answer hit rate     : 96.7%
  Table questions     : 96.4%
  Text questions      : 100.0%
  Avg latency         : 4.15s

Text RAG Baseline
  Retrieval accuracy  : 53.3%
  Answer hit rate     : 76.7%
  Table questions     : 78.6%
  Text questions      : 50.0%
  Avg latency         : 2.28s

★  Table question lift: Apertura beats text RAG by 17.8%
```

---

## Observability (Langfuse)

Add to `.env`:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Every pipeline run is traced with per agent latency, token cost, confidence score, and a direct link to the Langfuse dashboard. The UI shows a "View trace in Langfuse →" link on every answer, if the keys are added

---

## Production Deployment

**Docker Compose (full local stack with all microservices):**
```bash
docker compose -f docker-compose.full.yml up --build
```

**Kubernetes (local kind cluster):**
```bash
kind create cluster --config k8s/kind-config.yaml
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml -n apertura
kubectl apply -f k8s/ -n apertura
kubectl get pods -n apertura -w
```

**Lambda async ingestion:** See `lambda/ARCHITECTURE.md` for the async S3-triggered ingestion architecture that decouples PDF processing from the query path

---

## Project Structure

```
apertura/
├── src/apertura/
│   ├── agents/
│   │   ├── state.py          Shared state dataclass
│   │   ├── nodes.py          5 agent nodes (router/retriever/reranker/answerer/verifier)
│   │   ├── pipeline.py       LangGraph graph with retry edge
│   │   └── observability.py  Langfuse tracing
│   ├── api/
│   │   ├── main.py           FastAPI gateway (local + cloud mode)
│   │   ├── gateway_app.py    Microservice gateway proxy
│   │   ├── ingestion_app.py  Ingestion microservice
│   │   ├── query_app.py      Query microservice
│   │   └── static.py         Page image serving (local)
│   ├── answer/               Claude vision answerer
│   ├── eval/                 Eval harness + text-RAG baseline
│   ├── ingestion/            PDF render → ColQwen2.5 → Qdrant
│   └── router/               Query classifier (visual vs text)
├── frontend/                 Next.js UI (Vercel)
├── scripts/
│   ├── serve.py              Start local backend
│   ├── ingest.py             Index a PDF
│   ├── ask.py                CLI question answering
│   ├── run_eval.py           Full 30-question eval
│   └── run_ci_eval.py        Fast CI gate
├── eval/
│   ├── golden_set.yaml       30 labeled questions
│   └── results.json          Last eval run
├── docker/                   Dockerfiles per microservice
├── docker-compose.yml        Qdrant (local dev)
├── docker-compose.full.yml   Full microservices stack
├── k8s/                      Kubernetes manifests
├── lambda/                   AWS Lambda async ingestion
├── grafana/                  Grafana dashboard config
├── modal_app.py              Modal GPU functions (cloud deployment)
└── blog/post.md              Technical writeup
```

---

## Blog Post

Read the full technical writeup: [`blog/post.md`](blog/post.md)

Covers the visual retrieval architecture, the baseline study methodology, the transformers version compatibility issue that caused non deterministic retrieval, and lessons learned building a production RAG system with LangGraph and Langfuse.

---

## License

MIT
