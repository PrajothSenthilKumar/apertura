# I Built a Multimodal RAG System That Beats Text Extraction by 17.8% on Financial Table Questions

*How visual document retrieval with ColQwen2.5 and a LangGraph multi agent pipeline outperforms standard text based RAG on the content that actually matters in financial filings.*

---

## The Problem

Every financial analyst knows the pain. You open a 150 page 10-K looking for one number, let's say free cash flow by segment, and it's buried inside a nested table on page 87. Ctrl-F doesn't help because you don't know the exact phrase. Standard RAG doesn't help because text extraction destroys the table structure: a clean income statement becomes a wall of numbers with no alignment, no headers, no context.

I built Apertura to solve this. The core insight: instead of extracting text from PDFs and then embedding it, embed the *rendered page image* directly. The layout, the tables, the charts all of it becomes searchable.

The result: **96.7% answer accuracy on a 30 questions golden set, versus 76.7% for a text extraction baseline. A 17.8% lift on table and chart questions.**

---

## Why Visual Retrieval

The standard RAG recipe ---> PDF to OCR text to chunks to single vector embeddings that loses exactly the content that matters most in financial documents.

Consider a segment revenue table:

```
| Segment          | Q1 FY27    | Q1 FY26    | Y/Y   |
|------------------|------------|------------|-------|
| Data Center      | $75,246M   | $39,112M   | +92%  |
| Edge Computing   |  $6,369M   |  $4,950M   | +29%  |
```

PyMuPDF (the best in class text extractor) turns this into:
```
Data Center 75,246 39,112 21 % 92 % Edge Computing 6,369 4,950 10 % 29 %
```

The relationship between the number and its label is gone. When a language model tries to answer "what was Data Center revenue?", it's working from a string soup.

ColQwen2.5 never sees the text. It sees the rendered page image, which is the full visual layout, the column alignment, the bold headers. It embeds the image into a multi vector representation (one vector per image patch) and scores queries using late interaction MaxSim, the same algorithm as ColBERT. The right page rises to the top because ColQwen2.5 understands that "Data Center" and "$75,246M" are spatially related.

---

## Architecture

The system has five layers:

**Ingestion** --> PDF pages are rendered to JPEG images at 150 DPI using pdf2image. Each page image is embedded by ColQwen2.5 running on an NVIDIA GPU, producing a multi vector representation (one 128 dim vector per visual patch). These multi vectors are stored in Qdrant with a multi vector collection configured for MaxSim scoring.

**Query routing** --> Incoming questions are classified by Claude Haiku (fast, cheap) as "visual" (answer likely in a table or chart) or "text" (answer in prose). This routes the question to the appropriate retrieval path and tells the UI which badge to show.

**Visual retrieval** --> The question is embedded with ColQwen2.5 (same model as ingestion) and scored against all stored page vectors using MaxSim. The top 5 candidate pages are retrieved.

**Reranking** --> Candidates are deduplicated by page number and trimmed to the top 3. In production, a cross encoder model would re score here, but currently the ColQwen2.5 retrieval score is used directly.

**Answer + verification** --> The top 3 page images are sent to Claude's vision API alongside the question. Claude reads the tables and charts directly from the images and produces a grounded answer. A separate Verifier agent then checks whether the answer is supported by the pages and assigns a confidence score. If confidence is below 0.6, the pipeline loops back to retrieval and tries again (up to 2 retries).

This five node pipeline is implemented in LangGraph, which models it as a state machine with a conditional retry edge from the Verifier back to the Retriever.

---

## The Baseline Study

To make a credible claim that visual RAG beats text RAG, you need a controlled comparison on the same questions with the same checking methodology.

**Golden set:-** 30 questions from Apple's Q1 2026 Form 10-Q, manually labeled with the expected answer fragment and ground truth page. 27 of 30 questions target tables or charts (income statement, balance sheet, segment revenue, gross margin, cash flow) while 3 target prose (legal risk, product announcements).

**Answer hit rate:-** after the system answers, we check whether the expected figure appears in the answer string. For example, for "What were total net sales?", the expected fragment is "143,756". Commas are stripped before comparison so "$143,756 million" matches "143756".

**Retrieval accuracy:-** whether the ground truth page appears in the top 3 retrieved pages.

**Text RAG baseline:-** PyMuPDF extracts raw text from each page. Sentence transformers (all-MiniLM-L6-v2) embeds the text. Cosine similarity retrieves the top 3 text pages. Claude receives the extracted text (not images) and answers from it.

**Results:**

| Metric | Apertura (visual) | Text RAG baseline |
|---|---|---|
| Answer accuracy | **96.7%** | 76.7% |
| Table questions | **96.4%** | 78.6% |
| Text questions | 100% | 50% |
| Retrieval accuracy | **83.3%** | 53.3% |
| Avg latency | 4.15s | 2.28s |

The 17.8% lift on table questions is the headline. The retrieval accuracy gap (83.3% vs 53.3%) is arguably more fundamental that text baseline retrieves the wrong page almost half the time on structured content, so no amount of LLM capability can recover.

The latency trade off is real: Apertura takes ~2 extra seconds per query because page images carry more tokens than extracted text. The query router mitigates this by routing prose questions to a cheaper text path.

---

## What I Learned

**Multi vector indexing is non negotiable for ColPali style models.** ColQwen2.5 produces one vector per image patch, not one vector per page. Standard single vector databases (pgvector with a single embedding) force you to mean pool the patches, which loses the fine grained spatial information that makes late interaction work. Qdrant's multi-vector collection with MaxSim is the right tool.

**The version matrix matters more than I expected.** ColQwen2.5's LoRA adapter loads correctly only with `colpali-engine==0.3.9` and `transformers>=4.50.0,<4.51.0`. Newer transformers versions (5.x) renamed the internal module structure, causing the adapter to silently load as random weights and producing non-deterministic, wrong retrieval results that look like they might be working. Always verify determinism by running the same query twice and checking that the scores match exactly.

**Confidence gated verification catches the failure mode.** The Verifier catches the ~3% of cases where the Answerer produces a plausible sounding but unsupported answer. For financial data where a wrong figure could drive a real decision, "low confidence - please verify" is far better than a confident wrong answer.

**The eval harness is the project.** The 96.7% number wouldn't exist without the 30 question golden set and the text RAG baseline to compare against. Building the eval infrastructure first before tuning anything that is what separates a demo from a system you can actually trust.

---

## What's Next

- Expand the golden set to 100+ questions across multiple companies (Microsoft, NVIDIA, Tesla) to validate generalization
- Add a cross encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) for the reranking node
- Move ingestion to async Lambda so large filings process in the background
- Deploy on Modal.com for a public live demo without maintaining a 24/7 GPU server

---

## Code

Everything is on GitHub: [github.com/PrajothSenthilKumar/apertura](https://github.com/PrajothSenthilKumar/apertura)

The repo includes the ingestion pipeline, LangGraph agent code, eval harness with the golden set, Docker/K8s configs, and the Lambda architecture. PRs welcome.
