#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GEM TSV runs against SciDocs ground truth."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="Path to queries.jsonl",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus.jsonl",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        required=True,
        help="Path to SciDocs qrels TSV",
    )
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="Path to one GEM TSV run file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--runs-glob",
        type=str,
        help="Optional glob pattern for GEM TSV runs, e.g. 'example_index/scidocs_results_*.tsv'",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[10, 100],
        help="Metrics cutoffs to report.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save a summary CSV.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional GEM run log file to parse average query time and QPS.",
    )
    parser.add_argument(
        "--query-order",
        choices=["queries-jsonl", "qrels-first-seen"],
        default="queries-jsonl",
        help="Order used by numeric GEM query IDs.",
    )
    return parser.parse_args()


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


def load_corpus_ids(path: Path) -> list[str]:
    ids: list[str] = []
    failed = False
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                failed = True
                break
            ids.append(str(payload["_id"]))

    if not failed:
        return ids

    text = path.read_text(encoding="utf-8")
    regex_ids = re.findall(r'"_id"\s*:\s*"([^"]+)"', text)
    print(
        f"Warning: {path} is not clean JSONL; using regex fallback and extracted {len(regex_ids)} document IDs.",
        file=sys.stderr,
    )
    return regex_ids


def build_positive_qrels(
    qrels_path: Path, query_to_idx: dict[str, int], doc_to_idx: dict[str, int]
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    stats = {
        "positive_rows": 0,
        "missing_query_ids": 0,
        "missing_doc_ids": 0,
    }

    with qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = int(row["score"])
            if score <= 0:
                continue

            stats["positive_rows"] += 1
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])

            if query_id not in query_to_idx:
                stats["missing_query_ids"] += 1
                continue
            if corpus_id not in doc_to_idx:
                stats["missing_doc_ids"] += 1
                continue

            mapped_qid = str(query_to_idx[query_id])
            mapped_did = str(doc_to_idx[corpus_id])
            qrels[mapped_qid][mapped_did] = score

    return dict(qrels), stats


def load_positive_qrel_query_order(qrels_path: Path) -> list[str]:
    ordered: dict[str, None] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                ordered.setdefault(str(row["query-id"]), None)
    return list(ordered)


def load_run(path: Path) -> dict[str, list[tuple[int, str, float]]]:
    results: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_no, row in enumerate(reader, start=1):
            if len(row) == 4:
                query_id, doc_id, score, rank = row
            elif len(row) == 3:
                query_id, doc_id, score = row
                rank = len(results[str(query_id)]) + 1
            else:
                raise RuntimeError(f"Unexpected TSV row shape in {path} at line {row_no}: {row}")
            results[str(query_id)].append((int(rank), str(doc_id), float(score)))

    for query_id in results:
        results[query_id].sort(key=lambda item: item[0])
    return dict(results)


def dcg(relevances: list[int]) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances, start=1):
        total += (2**rel - 1) / math.log2(idx + 1)
    return total


def average_precision_at_k(top_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if not qrels:
        return 0.0

    score = 0.0
    hits = 0
    for rank, doc_id in enumerate(top_doc_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            hits += 1
            score += hits / rank
    return score / len(qrels)


def evaluate_run(
    qrels: dict[str, dict[str, int]],
    results: dict[str, list[tuple[int, str, float]]],
    k_values: list[int],
) -> dict[str, float]:
    common_qids = sorted(set(qrels) & set(results), key=int)
    summary: dict[str, float] = {
        "queries_with_qrels": float(len(qrels)),
        "queries_with_results": float(len(results)),
        "common_queries": float(len(common_qids)),
    }

    if not common_qids:
        return summary

    accumulators = {
        k: {"mrr": 0.0, "ndcg": 0.0, "recall": 0.0, "precision": 0.0, "map": 0.0}
        for k in k_values
    }

    for query_id in common_qids:
        ranked = results[query_id]
        qrels_for_query = qrels[query_id]
        doc_ids = [doc_id for _, doc_id, _ in ranked]
        num_rel = len(qrels_for_query)

        for k in k_values:
            top_doc_ids = doc_ids[:k]
            top_rels = [qrels_for_query.get(doc_id, 0) for doc_id in top_doc_ids]
            hits = sum(1 for doc_id in top_doc_ids if qrels_for_query.get(doc_id, 0) > 0)

            rr = 0.0
            for rank, doc_id in enumerate(top_doc_ids, start=1):
                if qrels_for_query.get(doc_id, 0) > 0:
                    rr = 1.0 / rank
                    break

            ideal_rels = [1] * min(num_rel, k)
            ideal_dcg = dcg(ideal_rels)
            ndcg = dcg(top_rels) / ideal_dcg if ideal_dcg > 0 else 0.0

            accumulators[k]["mrr"] += rr
            accumulators[k]["ndcg"] += ndcg
            accumulators[k]["recall"] += (hits / num_rel) if num_rel else 0.0
            accumulators[k]["precision"] += hits / k
            accumulators[k]["map"] += average_precision_at_k(top_doc_ids, qrels_for_query, k)

    query_count = len(common_qids)
    for k in k_values:
        summary[f"MRR@{k}"] = accumulators[k]["mrr"] / query_count
        summary[f"NDCG@{k}"] = accumulators[k]["ndcg"] / query_count
        summary[f"Recall@{k}"] = accumulators[k]["recall"] / query_count
        summary[f"P@{k}"] = accumulators[k]["precision"] / query_count
        summary[f"MAP@{k}"] = accumulators[k]["map"] / query_count

    return summary


def discover_runs(args: argparse.Namespace) -> list[Path]:
    run_paths = list(args.run)
    if args.runs_glob:
        run_paths.extend(sorted(Path(match) for match in glob.glob(args.runs_glob)))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in run_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def parse_run_key_from_name(path: Path) -> tuple[int, int | None, int] | None:
    match = re.search(r"(?:_t(\d+))?_rerank(\d+)_ef(\d+)\.tsv$", path.name)
    if not match:
        return None
    nprob = int(match.group(1)) if match.group(1) is not None else None
    return int(match.group(2)), nprob, int(match.group(3))


def load_qps_map_from_sidecars(run_paths: list[Path]) -> dict[tuple[int, int | None, int], float]:
    qps_map: dict[tuple[int, int | None, int], float] = {}
    for run_path in run_paths:
        run_key = parse_run_key_from_name(run_path)
        if run_key is None:
            continue
        meta_path = Path(str(run_path) + ".meta.json")
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        qps = payload.get("qps")
        if qps is not None:
            qps_map[run_key] = float(qps)
    return qps_map


def load_qps_map_from_log(path: Path | None) -> dict[tuple[int, int | None, int], float]:
    if path is None:
        return {}

    text = path.read_text(encoding="utf-8")
    rerank_matches = list(re.finditer(r"(?:t/NPROB:\s*(\d+)\s+)?rerankK:\s*(\d+)\s+ef:\s*(\d+)", text))
    time_matches = list(re.finditer(r"Average query time:\s*([0-9.]+)\s+seconds", text))
    pair_count = min(len(rerank_matches), len(time_matches))

    qps_map: dict[tuple[int, int | None, int], float] = {}
    for idx in range(pair_count):
        nprob = int(rerank_matches[idx].group(1)) if rerank_matches[idx].group(1) is not None else None
        rerank = int(rerank_matches[idx].group(2))
        ef = int(rerank_matches[idx].group(3))
        avg_query_time = float(time_matches[idx].group(1))
        if avg_query_time > 0:
            qps_map[(rerank, nprob, ef)] = 1.0 / avg_query_time
    return qps_map


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    run_paths = discover_runs(args)
    if not run_paths:
        raise RuntimeError("No run files provided. Use --run or --runs-glob.")

    query_ids = load_jsonl_ids(args.queries)
    if args.query_order == "qrels-first-seen":
        query_ids = load_positive_qrel_query_order(args.qrels)
    corpus_ids = load_corpus_ids(args.corpus)
    qps_map = load_qps_map_from_sidecars(run_paths)
    for run_key, qps in load_qps_map_from_log(args.log_file).items():
        qps_map[run_key] = qps
    query_to_idx = {query_id: idx for idx, query_id in enumerate(query_ids)}
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    qrels, qrel_stats = build_positive_qrels(args.qrels, query_to_idx, doc_to_idx)

    print(f"query_ids\t{len(query_ids)}")
    print(f"doc_ids\t{len(corpus_ids)}")
    print(f"positive_qrels\t{qrel_stats['positive_rows']}")
    print(f"mapped_positive_qrels\t{sum(len(v) for v in qrels.values())}")
    print(f"queries_with_positive_qrels\t{len(qrels)}")
    print(f"missing_positive_qrel_query_ids\t{qrel_stats['missing_query_ids']}")
    print(f"missing_positive_qrel_doc_ids\t{qrel_stats['missing_doc_ids']}")
    print()

    csv_rows: list[dict[str, str]] = []
    for run_path in run_paths:
        results = load_run(run_path)
        metrics = evaluate_run(qrels, results, args.k_values)
        run_key = parse_run_key_from_name(run_path)
        if run_key is not None and run_key in qps_map:
            metrics["QPS"] = qps_map[run_key]
        if run_key is not None and run_key[1] is not None:
            metrics["t"] = float(run_key[1])

        print(run_path)
        for key, value in metrics.items():
            if key.startswith(("queries_", "common_queries")):
                print(f"{key}\t{int(value)}")
            else:
                print(f"{key}\t{value:.6f}")
        print()

        row: dict[str, str] = {"run": str(run_path)}
        for key, value in metrics.items():
            if key.startswith(("queries_", "common_queries")):
                row[key] = str(int(value))
            else:
                row[key] = f"{value:.6f}"
        csv_rows.append(row)

    if args.output_csv:
        write_summary_csv(args.output_csv, csv_rows)
        print(f"Saved summary CSV to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
