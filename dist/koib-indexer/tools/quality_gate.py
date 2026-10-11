"""Repeatable retrieval quality gate on a domain-specific JSONL benchmark.

Benchmark rows:
  {"query":"...", "source":"subdir/manual.pdf", "contains":"required value"}

Does not call an LLM or claim factual answer accuracy. ``source`` and ``contains``
are optional independently. A hit must satisfy all supplied conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_cases(retriever, rows: list[dict], k: int = 5) -> dict:
    if not rows or k < 1:
        raise ValueError("Nonempty questions and k>=1 required")
    reciprocal = 0.0
    hits = 0
    details = []
    for row in rows:
        if not row.get("query") or not (row.get("source") or row.get("contains")):
            raise ValueError("Each case needs 'query' and 'source' or 'contains'")
        found = retriever.search(row["query"], k=k, use_hyde=False)
        rank = 0
        for i, r in enumerate(found, 1):
            data = "\n".join([r.content or "", r.full_content or ""])
            source_ok = not row.get("source") or r.source == row["source"]
            content_ok = not row.get("contains") or row["contains"].casefold() in data.casefold()
            if source_ok and content_ok:
                rank = i
                break
        hits += bool(rank)
        reciprocal += 1 / rank if rank else 0
        details.append({"query": row["query"], "hit": bool(rank), "rank": rank,
                        "expected_source": row.get("source"), "expected_contains": row.get("contains")})
    n = len(rows)
    return {"total": n, "k": k, "recall_at_k": hits / n,
            "mrr_at_k": reciprocal / n, "failures": n - hits, "cases": details}


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality regression gate for a real document corpus")
    ap.add_argument("--cases", type=Path, required=True, help="JSONL queries with expected source/substring")
    ap.add_argument("--output-dir", type=Path, default=Path("output"), help="root with index/ and docstore/")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-recall", type=float, default=0.85)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not 0 <= args.min_recall <= 1:
        ap.error("--min-recall must be between 0 and 1")
    rows = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    from src.retrieval import HybridRetriever
    from src.indexing import IndexBuilder
    if not (args.output_dir / "index" / "bm25_fts.db").is_file():
        raise FileNotFoundError(f"Index not found in {args.output_dir}")
    index = IndexBuilder(index_dir=args.output_dir / "index",
                         docstore_path=args.output_dir / "docstore" / "docstore.db")
    retriever = HybridRetriever(index)
    try:
        report = score_cases(retriever, rows, k=args.k)
    finally:
        retriever.close()
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    if report["recall_at_k"] < args.min_recall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
