"""DeepEval CI gate.

Runs a fast subset of the golden set through the visual RAG pipeline and
fails if answer hit rate drops below MIN_ANSWER_HIT_RATE.

Run locally:   python scripts/run_ci_eval.py --pdf 10QQ12026.pdf
Run in CI:     add this as a step after pip install -e .

Fails with exit code 1 if quality regresses — blocks the PR.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from apertura.answer.answerer import answer_question
from apertura.ingestion.embedder import Embedder
from apertura.ingestion.pipeline import ingest_pdf
from apertura.ingestion.vector_store import VectorStore

MIN_ANSWER_HIT_RATE = 0.80   # fail CI if below 80%
CI_QUESTION_IDS = {          # run only these 10 for speed in CI
    "q01", "q02", "q07", "q09", "q11",
    "q14", "q17", "q21", "q24", "q28",
}


def answer_hit(answer: str, fragment: str) -> bool:
    return fragment.lower().replace(",", "") in answer.lower().replace(",", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--doc-id", default="ci-eval")
    parser.add_argument("--golden", default="eval/golden_set.yaml")
    args = parser.parse_args()

    with open(args.golden) as f:
        all_questions = yaml.safe_load(f)["questions"]

    questions = [q for q in all_questions if q["id"] in CI_QUESTION_IDS]
    print(f"\nRunning CI eval on {len(questions)} questions …")

    embedder = Embedder()
    store = VectorStore()
    store.ensure_collection()

    if store.client.get_collection(store.collection).points_count == 0:
        print("Ingesting …")
        ingest_pdf(args.pdf, doc_id=args.doc_id, embedder=embedder, store=store)

    hits = 0
    for q in questions:
        query_vec = embedder.embed_query(q["question"])
        pages = store.search(query_vec, limit=3)
        page_paths = [p.payload["image_path"] for p in pages]
        answer = answer_question(q["question"], page_paths)
        ok = answer_hit(answer, q["expected_fragment"])
        hits += ok
        print(f"  [{'✓' if ok else '✗'}] {q['id']}")

    rate = hits / len(questions)
    print(f"\nAnswer hit rate: {rate:.1%} (threshold: {MIN_ANSWER_HIT_RATE:.1%})")

    if rate < MIN_ANSWER_HIT_RATE:
        print(f"❌ CI FAILED — quality regressed below {MIN_ANSWER_HIT_RATE:.1%}")
        sys.exit(1)
    else:
        print("✅ CI PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
