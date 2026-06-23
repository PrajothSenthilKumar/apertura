import argparse
import json
import time
from pathlib import Path
import yaml
from apertura.answer.answerer import answer_question
from apertura.config import get_settings
from apertura.eval.text_baseline import TextBaseline
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.pipeline import ingest_pdf
from apertura.ingestion.vector_store import VectorStore

def load_golden(path="eval/golden_set.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)["questions"]

def retrieval_hit(hits, ground_truth_page):
    return any(
        h.payload.get("page_num") == ground_truth_page if hasattr(h, "payload")
        else h.get("page_num") == ground_truth_page
        for h in hits
    )

def answer_hit(answer, fragment):
    return fragment.lower().replace(",", "") in answer.lower().replace(",", "")

def run_visual_rag(questions, embedder, store, top_k=3):
    results = []
    for q in questions:
        t0 = time.time()
        query_vec = embedder.embed_query(q["question"])
        hits = store.search(query_vec, limit=top_k)
        page_paths = [h.payload["image_path"] for h in hits]
        page_nums = [h.payload["page_num"] for h in hits]
        answer = answer_question(q["question"], page_paths)
        elapsed = round(time.time() - t0, 2)
        ret_hit = retrieval_hit(hits, q["ground_truth_page"])
        ans_hit = answer_hit(answer, q["expected_fragment"])
        results.append({"id": q["id"], "question": q["question"], "question_type": q["question_type"],
            "ground_truth_page": q["ground_truth_page"], "retrieved_pages": page_nums,
            "retrieval_hit": ret_hit, "answer": answer, "answer_hit": ans_hit, "latency_s": elapsed})
        print(f"  [{'✓' if ans_hit else '✗'}] {q['id']} — pages {page_nums} — {elapsed}s")
    return results

def run_text_baseline(questions, baseline, top_k=3):
    from anthropic import Anthropic
    settings = get_settings()
    client = Anthropic(api_key=settings.anthropic_api_key)
    results = []
    for q in questions:
        t0 = time.time()
        hits = baseline.search(q["question"], k=top_k)
        context = "\n\n---\n\n".join(h["text"][:2000] for h in hits)
        page_nums = [h["page_num"] for h in hits]
        resp = client.messages.create(
            model=settings.answer_model, max_tokens=512,
            system="You are a financial document analyst. Answer using ONLY the provided text context. Be concise and quote exact figures.",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q['question']}"}],
        )
        answer = "".join(b.text for b in resp.content if b.type == "text")
        elapsed = round(time.time() - t0, 2)
        ret_hit = retrieval_hit(hits, q["ground_truth_page"])
        ans_hit = answer_hit(answer, q["expected_fragment"])
        results.append({"id": q["id"], "question": q["question"], "question_type": q["question_type"],
            "ground_truth_page": q["ground_truth_page"], "retrieved_pages": page_nums,
            "retrieval_hit": ret_hit, "answer": answer, "answer_hit": ans_hit, "latency_s": elapsed})
        print(f"  [{'✓' if ans_hit else '✗'}] {q['id']} — pages {page_nums} — {elapsed}s")
    return results

def summarise(results, label):
    total = len(results)
    table_qs = [r for r in results if r["question_type"] == "table"]
    text_qs = [r for r in results if r["question_type"] == "text"]
    return {
        "system": label,
        "total_questions": total,
        "retrieval_accuracy": round(sum(r["retrieval_hit"] for r in results) / total, 3),
        "answer_hit_rate": round(sum(r["answer_hit"] for r in results) / total, 3),
        "table_answer_hit_rate": round(sum(r["answer_hit"] for r in table_qs) / max(len(table_qs), 1), 3),
        "text_answer_hit_rate": round(sum(r["answer_hit"] for r in text_qs) / max(len(text_qs), 1), 3),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / total, 2),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--doc-id", default="eval-doc")
    parser.add_argument("--golden", default="eval/golden_set.yaml")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    questions = load_golden(args.golden)
    print(f"\nLoaded {len(questions)} golden questions")
    output = {"visual_rag": [], "text_baseline": [], "summaries": []}

    if not args.skip_visual:
        print("\n── Visual RAG (Apertura) ─────────────────────────────")
        embedder = Embedder()
        store = VectorStore()
        store.ensure_collection()
        stats = store.client.get_collection(store.collection)
        if stats.points_count == 0:
            print(f"Ingesting {args.pdf} …")
            ingest_pdf(args.pdf, doc_id=args.doc_id, embedder=embedder, store=store)
        output["visual_rag"] = run_visual_rag(questions, embedder, store)
        output["summaries"].append(summarise(output["visual_rag"], "Apertura Visual RAG"))

    if not args.skip_baseline:
        print("\n── Text-RAG Baseline ─────────────────────────────────")
        baseline = TextBaseline()
        n = baseline.index_pdf(args.pdf, doc_id=args.doc_id)
        print(f"Indexed {n} pages")
        output["text_baseline"] = run_text_baseline(questions, baseline)
        output["summaries"].append(summarise(output["text_baseline"], "Text-RAG Baseline"))

    print("\n══ RESULTS ══════════════════════════════════════════════")
    for s in output["summaries"]:
        print(f"\n{s['system']}")
        print(f"  Retrieval accuracy  : {s['retrieval_accuracy']:.1%}")
        print(f"  Answer hit rate     : {s['answer_hit_rate']:.1%}")
        print(f"  Table questions     : {s['table_answer_hit_rate']:.1%}")
        print(f"  Text questions      : {s['text_answer_hit_rate']:.1%}")
        print(f"  Avg latency         : {s['avg_latency_s']}s")

    if len(output["summaries"]) == 2:
        vis = output["summaries"][0]
        txt = output["summaries"][1]
        lift = vis["table_answer_hit_rate"] - txt["table_answer_hit_rate"]
        print(f"\n★  Table-question lift: Apertura beats text RAG by {lift:.1%} on table/chart questions")

    Path("eval/results.json").write_text(json.dumps(output, indent=2))
    print("\nFull results saved to eval/results.json")

if __name__ == "__main__":
    main()