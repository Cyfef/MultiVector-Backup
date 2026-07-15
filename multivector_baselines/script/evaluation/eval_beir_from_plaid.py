import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from beir.retrieval.evaluation import EvaluateRetrieval


def load_ordered_ids(jsonl_path: Path) -> List[str]:
    ids = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            if "_id" not in record:
                raise KeyError(f"{jsonl_path} line {line_no} does not contain '_id'")
            ids.append(str(record["_id"]))
    return ids


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if row_idx == 0 and row[0] == "query-id":
                continue
            if len(row) < 3:
                raise ValueError(f"Malformed qrels row in {qrels_path}: {row}")
            query_id, corpus_id, score = str(row[0]), str(row[1]), int(float(row[2]))
            qrels.setdefault(query_id, {})[corpus_id] = score
    return qrels


def infer_performance_path(answer_path: Path, performance_dir: Path) -> Path:
    performance_name = answer_path.name.replace("-plaid-top", "-retrieval-plaid-top", 1).replace(".tsv", ".json")
    return performance_dir / performance_name


def parse_answer_file(
    answer_path: Path,
    query_ids: List[str],
    corpus_ids: List[str],
) -> Tuple[Dict[str, Dict[str, float]], int, int, int]:
    results: Dict[str, Dict[str, float]] = {}
    max_rank = 0
    max_local_qid = -1
    line_count = 0

    with answer_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"Malformed answer row in {answer_path}: {row}")

            local_qid = int(row[0])
            local_docid = int(row[1])
            rank = int(row[2])
            score = float(row[3])

            if local_qid >= len(query_ids) or local_qid < 0:
                raise IndexError(f"Query index {local_qid} out of range for {answer_path}")
            if local_docid >= len(corpus_ids) or local_docid < 0:
                raise IndexError(f"Corpus index {local_docid} out of range for {answer_path}")

            query_id = query_ids[local_qid]
            corpus_id = corpus_ids[local_docid]
            results.setdefault(query_id, {})[corpus_id] = score

            max_local_qid = max(max_local_qid, local_qid)
            max_rank = max(max_rank, rank)
            line_count += 1

    n_result_queries = len(results)
    if max_local_qid + 1 != n_result_queries:
        # The file may still be valid if some queries have no retrieved items, but
        # for the current Plaid output format each query should appear at least once.
        raise ValueError(
            f"{answer_path} does not contain a dense set of query ids: "
            f"max_local_qid={max_local_qid}, distinct_queries={n_result_queries}"
        )

    return results, max_rank, line_count, n_result_queries


def to_float(value):
    if value is None:
        return None
    return float(value)


def load_performance(performance_path: Path) -> dict:
    with performance_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metric_dict: dict, prefixes: Iterable[str], k: int):
    for prefix in prefixes:
        key = f"{prefix}@{k}"
        if key in metric_dict:
            return metric_dict[key]
    return None


def evaluate_answer_file(
    answer_path: Path,
    performance_path: Path,
    qrels: Dict[str, Dict[str, int]],
    query_ids: List[str],
    corpus_ids: List[str],
    k_values: List[int],
    per_run_dir: Path,
) -> dict:
    results, max_rank, line_count, n_result_queries = parse_answer_file(
        answer_path=answer_path,
        query_ids=query_ids,
        corpus_ids=corpus_ids,
    )

    performance = load_performance(performance_path)

    for query_id in qrels:
        results.setdefault(query_id, {})

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="mrr")

    search_time = performance["search_time"]
    total_query_time_ms = float(search_time["total_query_time_ms"])
    n_query = int(performance["n_query"])
    qps = (1000.0 * n_query / total_query_time_ms) if total_query_time_ms > 0 else None

    row = {
        "answer_file": answer_path.name,
        "performance_file": performance_path.name,
        "n_query": n_query,
        "retrieved_queries": n_result_queries,
        "result_lines": line_count,
        "requested_topk": int(performance["topk"]),
        "observed_max_rank": max_rank,
        "ndocs": performance["retrieval"]["ndocs"],
        "ncells": performance["retrieval"]["ncells"],
        "centroid_score_threshold": performance["retrieval"]["centroid_score_threshold"],
        "n_thread": performance["retrieval"]["n_thread"],
        "qps": qps,
        "total_query_time_ms": total_query_time_ms,
        "average_query_time_ms": to_float(search_time["average_query_time_ms"]),
        "retrieval_time_p5_ms": to_float(search_time["retrieval_time_p5(ms)"]),
        "retrieval_time_p50_ms": to_float(search_time["retrieval_time_p50(ms)"]),
        "retrieval_time_p95_ms": to_float(search_time["retrieval_time_p95(ms)"]),
        "average_ivf_time_ms": to_float(search_time["average_ivf_time_ms"]),
        "average_filter_time_ms": to_float(search_time["average_filter_time_ms"]),
        "average_refine_time_ms": to_float(search_time["average_refine_time_ms"]),
        "average_n_refine_ivf": to_float(search_time["average_n_refine_ivf"]),
        "average_n_refine_filter": to_float(search_time["average_n_refine_filter"]),
        "average_n_vec_score_refine": to_float(search_time["average_n_vec_score_refine"]),
    }

    per_run_metrics = {
        "answer_file": answer_path.name,
        "performance_file": performance_path.name,
        "requested_topk": row["requested_topk"],
        "observed_max_rank": max_rank,
        "retrieval": performance["retrieval"],
        "qps": qps,
        "search_time": search_time,
        "metrics": {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        },
    }
    per_run_path = per_run_dir / f"{answer_path.stem}.json"
    with per_run_path.open("w", encoding="utf-8") as f:
        json.dump(per_run_metrics, f, indent=2)

    for k in k_values:
        row[f"ndcg@{k}"] = metric_value(ndcg, ("NDCG",), k)
        row[f"map@{k}"] = metric_value(_map, ("MAP",), k)
        row[f"recall@{k}"] = metric_value(recall, ("Recall",), k)
        row[f"precision@{k}"] = metric_value(precision, ("P", "Precision"), k)
        row[f"mrr@{k}"] = metric_value(mrr, ("MRR",), k)

    return row


def write_summary(rows: List[dict], summary_path: Path, k_values: List[int]) -> None:
    base_fields = [
        "answer_file",
        "performance_file",
        "n_query",
        "retrieved_queries",
        "result_lines",
        "requested_topk",
        "observed_max_rank",
        "ndocs",
        "ncells",
        "centroid_score_threshold",
        "n_thread",
        "qps",
        "total_query_time_ms",
        "average_query_time_ms",
        "retrieval_time_p5_ms",
        "retrieval_time_p50_ms",
        "retrieval_time_p95_ms",
        "average_ivf_time_ms",
        "average_filter_time_ms",
        "average_refine_time_ms",
        "average_n_refine_ivf",
        "average_n_refine_filter",
        "average_n_vec_score_refine",
    ]
    metric_fields = []
    for metric_name in ("ndcg", "map", "recall", "precision", "mrr"):
        for k in k_values:
            metric_fields.append(f"{metric_name}@{k}")

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + metric_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_key(row: dict):
    return (
        int(row["requested_topk"]),
        int(row["ndocs"]),
        int(row["ncells"]),
        float(row["centroid_score_threshold"]),
        int(row["n_thread"]),
        row["answer_file"],
    )


def resolve_answer_files(answer_dir: Path, answer_glob: str, performance_dir: Path) -> List[Tuple[Path, Path]]:
    answer_paths = sorted(answer_dir.glob(answer_glob))
    pairs = []
    for answer_path in answer_paths:
        performance_path = infer_performance_path(answer_path, performance_dir)
        if performance_path.exists():
            pairs.append((answer_path, performance_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluate Plaid TSV outputs with BEIR metrics.")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="BEIR dataset directory containing corpus.jsonl, queries.jsonl, and qrels/<split>.tsv")
    parser.add_argument("--answer-dir", type=Path, required=True,
                        help="Directory containing Plaid answer TSV files")
    parser.add_argument("--performance-dir", type=Path, required=True,
                        help="Directory containing Plaid retrieval performance JSON files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for BEIR per-run JSON files and the summary CSV")
    parser.add_argument("--split", type=str, default="test", help="Qrels split to load, e.g. test or dev")
    parser.add_argument("--answer-glob", type=str, default="*.tsv",
                        help="Glob for answer TSVs inside --answer-dir")
    parser.add_argument("--summary-name", type=str, default="beir_summary.csv",
                        help="Filename for the aggregate CSV summary")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10, 100],
                        help="Cutoffs to evaluate with BEIR")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = args.output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    query_ids = load_ordered_ids(args.dataset_dir / "queries.jsonl")
    corpus_ids = load_ordered_ids(args.dataset_dir / "corpus.jsonl")
    qrels = load_qrels(args.dataset_dir / "qrels" / f"{args.split}.tsv")

    answer_pairs = resolve_answer_files(
        answer_dir=args.answer_dir,
        answer_glob=args.answer_glob,
        performance_dir=args.performance_dir,
    )
    if not answer_pairs:
        raise FileNotFoundError(
            f"No answer files matching {args.answer_glob} in {args.answer_dir} with corresponding performance JSONs"
        )

    rows = []
    for answer_path, performance_path in answer_pairs:
        row = evaluate_answer_file(
            answer_path=answer_path,
            performance_path=performance_path,
            qrels=qrels,
            query_ids=query_ids,
            corpus_ids=corpus_ids,
            k_values=args.k_values,
            per_run_dir=per_run_dir,
        )
        rows.append(row)

    rows.sort(key=sort_key)
    write_summary(rows=rows, summary_path=args.output_dir / args.summary_name, k_values=args.k_values)

    print(f"evaluated {len(rows)} Plaid result files")
    print(f"summary csv: {args.output_dir / args.summary_name}")
    print(f"per-run json dir: {per_run_dir}")


if __name__ == "__main__":
    main()
