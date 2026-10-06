"""
Evaluate retrieval accuracy of the Campus RAG Assistant against a labeled
test set.

Metrics computed:
- Hit@K: fraction of questions where the expected source appears anywhere
  in the top-K retrieved chunks.
- MRR (Mean Reciprocal Rank): average of 1/rank of the first chunk from the
  expected source, across all questions (0 if not found in top-K).

Usage (run from the `eval/` directory, with the backend's venv active and
notes already ingested via the API or ingest_notes.py):

    python evaluate.py
    python evaluate.py --k 5
    python evaluate.py --test-set test_set.json
"""

import argparse
import json
import sys
from pathlib import Path

# Allow importing the backend package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import rag_chain  # noqa: E402


def load_test_set(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


def evaluate(test_cases: list[dict], k: int = 4) -> dict:
    results = []
    hits = 0
    reciprocal_ranks = []

    for case in test_cases:
        question = case["question"]
        expected_source = case["expected_source"]

        retrieved = rag_chain.retrieve(question, k=k)
        retrieved_sources = [doc.metadata.get("source") for doc in retrieved]

        rank = None
        for i, src in enumerate(retrieved_sources, start=1):
            if src == expected_source:
                rank = i
                break

        is_hit = rank is not None
        hits += int(is_hit)
        reciprocal_ranks.append(1 / rank if is_hit else 0)

        results.append(
            {
                "id": case["id"],
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "hit": is_hit,
                "rank": rank,
            }
        )

    n = len(test_cases)
    summary = {
        "num_questions": n,
        "k": k,
        "hit_at_k": round(hits / n, 3) if n else 0.0,
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0.0,
    }
    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval accuracy.")
    parser.add_argument(
        "--test-set",
        default=str(Path(__file__).resolve().parent / "test_set.json"),
        help="Path to labeled test set JSON file.",
    )
    parser.add_argument("--k", type=int, default=4, help="Top-K chunks to retrieve.")
    args = parser.parse_args()

    test_cases = load_test_set(args.test_set)
    report = evaluate(test_cases, k=args.k)

    print(f"\n=== Retrieval Evaluation (k={report['summary']['k']}) ===")
    print(f"Questions evaluated : {report['summary']['num_questions']}")
    print(f"Hit@{report['summary']['k']}             : {report['summary']['hit_at_k']}")
    print(f"MRR                 : {report['summary']['mrr']}\n")

    print("--- Per-question detail ---")
    for r in report["details"]:
        status = "✅ HIT" if r["hit"] else "❌ MISS"
        rank_str = f"(rank {r['rank']})" if r["hit"] else ""
        print(f"[{status}] {rank_str} {r['id']}: {r['question']}")
        print(f"   expected : {r['expected_source']}")
        print(f"   retrieved: {r['retrieved_sources']}")

    out_path = Path(__file__).resolve().parent / "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
