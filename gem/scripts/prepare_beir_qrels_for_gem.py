#!/usr/bin/env python3
import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path


def load_jsonl_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            ids.append(str(payload["_id"]))
    return ids


def positive_qrels_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if int(row["score"]) <= 0:
                continue
            rows.append((str(row["query-id"]), str(row["corpus-id"])))
    return rows


def query_ids_from_qrels(rows: list[tuple[str, str]]) -> list[str]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for query_id, _ in rows:
        ordered.setdefault(query_id, None)
    return list(ordered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map BEIR qrels IDs to GEM's zero-based numeric qid/pid format.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--query-order",
        choices=["queries-jsonl", "qrels-first-seen"],
        default="queries-jsonl",
        help="Use queries.jsonl order, or the first-seen positive query order in qrels for filtered test-query embeddings.",
    )
    parser.add_argument("--expected-query-count", type=int, help="Fail if the mapped query count differs.")
    parser.add_argument("--expected-doc-count", type=int, help="Fail if the corpus count differs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = positive_qrels_rows(args.qrels)
    corpus_ids = load_jsonl_ids(args.corpus)
    query_ids = load_jsonl_ids(args.queries)
    if args.query_order == "qrels-first-seen":
        query_ids = query_ids_from_qrels(rows)

    if args.expected_query_count is not None and len(query_ids) != args.expected_query_count:
        raise RuntimeError(f"Query count mismatch: mapped={len(query_ids)} expected={args.expected_query_count}")
    if args.expected_doc_count is not None and len(corpus_ids) != args.expected_doc_count:
        raise RuntimeError(f"Corpus count mismatch: mapped={len(corpus_ids)} expected={args.expected_doc_count}")

    query_to_idx = {query_id: idx for idx, query_id in enumerate(query_ids)}
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}

    kept = 0
    skipped_query = 0
    skipped_doc = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for query_id, doc_id in rows:
            if query_id not in query_to_idx:
                skipped_query += 1
                continue
            if doc_id not in doc_to_idx:
                skipped_doc += 1
                continue
            out.write(f"{query_to_idx[query_id]}\t{doc_to_idx[doc_id]}\n")
            kept += 1

    print(f"wrote {args.output}")
    print(f"query_ids={len(query_ids)} corpus_ids={len(corpus_ids)} positive_rows={len(rows)} kept={kept}")
    print(f"skipped_missing_query={skipped_query} skipped_missing_doc={skipped_doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
