#!/usr/bin/env python3
"""Evaluate EMVB ranking files against BEIR qrels.

The EMVB binary writes positional query/document ids. This script maps those
positions back to the BEIR ids using the same GenericDataLoader split ordering,
then runs the standard BEIR evaluation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from beir.retrieval.evaluation import EvaluateRetrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to the BEIR dataset directory.")
    parser.add_argument("--split", default="test", help="BEIR split name, e.g. test/dev/train.")
    parser.add_argument("--run", type=Path, required=True, help="Path to EMVB TSV run file.")
    parser.add_argument(
        "--query-ids-file",
        type=Path,
        help="Optional ordered query-id file to use instead of dataset queries.jsonl.",
    )
    parser.add_argument(
        "--corpus-ids-file",
        type=Path,
        help="Optional ordered corpus-id file to use instead of dataset corpus.jsonl.",
    )
    parser.add_argument(
        "--query-id-mode",
        choices=("auto", "positional", "direct"),
        default="auto",
        help="How to interpret the run-file query column.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="Cutoffs to report.",
    )
    parser.add_argument("--run-log", type=Path, help="Optional perf_emvb stdout log to parse timing from.")
    parser.add_argument(
        "--avg-query-time-ns",
        type=float,
        help="Optional average query time in nanoseconds. Overrides --run-log timing if both are provided.",
    )
    parser.add_argument(
        "--restrict-to-run-queries",
        action="store_true",
        help="Evaluate only on the queries that appear in the run file. Useful for query subsets.",
    )
    parser.add_argument("--search-threads", type=int, default=1, help="Search thread count to record in outputs.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--output-csv", type=Path, help="Optional one-row CSV summary path.")
    return parser.parse_args()


def infer_query_id_mode(run_path: Path, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    unique_qids: list[int] = []
    seen: set[int] = set()
    with run_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_no, row in enumerate(reader, start=1):
            if len(row) != 4:
                raise RuntimeError(f"Unexpected row shape at line {row_no}: {row}")
            try:
                qid = int(row[0])
            except ValueError:
                return "direct"
            if qid not in seen:
                seen.add(qid)
                unique_qids.append(qid)
                if len(unique_qids) >= 256:
                    break

    if not unique_qids:
        return "positional"

    unique_qids.sort()
    if unique_qids[0] == 0 and unique_qids[-1] == len(unique_qids) - 1:
        return "positional"
    return "direct"


def load_emvb_results(
    run_path: Path,
    query_ids: list[str],
    corpus_ids: list[str],
    query_id_mode: str,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    resolved_query_id_mode = infer_query_id_mode(run_path, query_id_mode)
    with run_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_no, row in enumerate(reader, start=1):
            if len(row) != 4:
                raise RuntimeError(f"Unexpected row shape at line {row_no}: {row}")

            qid_raw, d_idx_raw, _rank_raw, score_raw = row
            d_idx = int(d_idx_raw)
            score = float(score_raw)

            if d_idx < 0 or d_idx >= len(corpus_ids):
                raise IndexError(f"Document index {d_idx} out of range at line {row_no}")

            if resolved_query_id_mode == "positional":
                q_idx = int(qid_raw)
                if q_idx < 0 or q_idx >= len(query_ids):
                    raise IndexError(f"Query index {q_idx} out of range at line {row_no}")
                qid = query_ids[q_idx]
            else:
                qid = qid_raw
            did = corpus_ids[d_idx]
            results.setdefault(qid, {})[did] = score

    return results


def extract_jsonl_id(line: str) -> str:
    prefix = '{"_id": "'
    if line.startswith(prefix):
        end = line.find('"', len(prefix))
        if end != -1:
            return line[len(prefix) : end]
    return json.loads(line)["_id"]


def load_ordered_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(extract_jsonl_id(line))
    return ids


def load_plain_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if item:
                ids.append(item)
    return ids


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(float(row["score"]))
            qrels.setdefault(qid, {})[did] = score
    return qrels


def resolve_timing(avg_query_time_ns: float | None, run_log: Path | None) -> dict[str, float]:
    if avg_query_time_ns is None and run_log is not None and run_log.exists():
        for line in run_log.read_text(encoding="utf-8").splitlines():
            prefix = "Average Elapsed Time per query"
            if line.startswith(prefix):
                avg_query_time_ns = float(line.split()[-1])

    if avg_query_time_ns is None:
        return {}

    avg_query_time_s = avg_query_time_ns / 1_000_000_000.0
    return {
        "avg_query_time_ns": avg_query_time_ns,
        "avg_query_time_s": avg_query_time_s,
        "qps": (1.0 / avg_query_time_s) if avg_query_time_s > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()

    corpus_ids = (
        load_plain_ids(args.corpus_ids_file)
        if args.corpus_ids_file is not None
        else load_ordered_ids(args.dataset_dir / "corpus.jsonl")
    )
    query_ids = (
        load_plain_ids(args.query_ids_file)
        if args.query_ids_file is not None
        else load_ordered_ids(args.dataset_dir / "queries.jsonl")
    )
    qrels = load_qrels(args.dataset_dir / "qrels" / f"{args.split}.tsv")
    results = load_emvb_results(
        args.run,
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        query_id_mode=args.query_id_mode,
    )

    if args.restrict_to_run_queries:
        selected_qids = [qid for qid in query_ids if qid in results and qid in qrels]
        qrels = {qid: qrels[qid] for qid in selected_qids}
        results = {qid: results[qid] for qid in selected_qids}
        query_ids = selected_qids

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, args.k_values)
    overlapping_qids = [qid for qid in results if qid in qrels]
    if overlapping_qids:
        qrels_for_mrr = {qid: qrels[qid] for qid in overlapping_qids}
        results_for_mrr = {qid: results[qid] for qid in overlapping_qids}
        mrr = EvaluateRetrieval.evaluate_custom(qrels_for_mrr, results_for_mrr, args.k_values, metric="mrr")
    else:
        mrr = {f"MRR@{k}": 0.0 for k in args.k_values}
    timing = resolve_timing(args.avg_query_time_ns, args.run_log)

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": args.split,
        "run": str(args.run.resolve()),
        "query_count": len(query_ids),
        "corpus_count": len(corpus_ids),
        "results_query_count": len(results),
        "search_threads": args.search_threads,
        "k_values": args.k_values,
        "ndcg": ndcg,
        "map": _map,
        "mrr": mrr,
        "recall": recall,
        "precision": precision,
    }
    summary.update(timing)

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.output_csv:
        flat_summary: dict[str, object] = {
            "dataset_dir": summary["dataset_dir"],
            "split": summary["split"],
            "run": summary["run"],
            "query_count": summary["query_count"],
            "corpus_count": summary["corpus_count"],
            "results_query_count": summary["results_query_count"],
            "search_threads": summary["search_threads"],
        }
        for key in ("avg_query_time_ns", "avg_query_time_s", "qps"):
            if key in summary:
                flat_summary[key] = summary[key]
        for section in ("ndcg", "map", "mrr", "recall", "precision"):
            for key, value in summary[section].items():
                flat_summary[key] = value

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_summary.keys()))
            writer.writeheader()
            writer.writerow(flat_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
