import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def read_groundtruth(path: Path) -> dict[int, list[int]]:
    groundtruth = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            if "qid" not in record or "answer_pids" not in record:
                raise KeyError(f"{path} line {line_no} must contain qid and answer_pids")
            groundtruth[int(record["qid"])] = [int(pid) for pid in record["answer_pids"]]
    return groundtruth


def parse_answer(path: Path) -> dict[int, list[int]]:
    results = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"Malformed result row in {path}: {row}")
            qid = int(row[0])
            pid = int(row[1])
            rank = int(row[2])
            results.setdefault(qid, [])
            if len(results[qid]) < rank:
                results[qid].append(pid)
    return results


def infer_performance_path(answer_path: Path, performance_dir: Path) -> Path:
    performance_name = answer_path.name.replace("-top", "-retrieval-", 1).replace(".tsv", ".json")
    if "-plaid-" in answer_path.name:
        performance_name = answer_path.name.replace("-plaid-top", "-retrieval-plaid-top", 1).replace(".tsv", ".json")
    if "-dessert-" in answer_path.name:
        performance_name = answer_path.name.replace("-dessert-top", "-retrieval-dessert-top", 1).replace(".tsv", ".json")
    if "-MUVERA-" in answer_path.name:
        performance_name = answer_path.name.replace("-MUVERA-top", "-retrieval-MUVERA-top", 1).replace(".tsv", ".json")
    if "-IGP-" in answer_path.name:
        performance_name = answer_path.name.replace("-IGP-top", "-retrieval-IGP-top", 1).replace(".tsv", ".json")
    return performance_dir / performance_name

def get_performance_path(answer_path: Path, answer_base: Path, performance_base: Path) -> Path:
    """
    根据答案文件路径和基础目录，推断性能文件路径。
    假设答案文件和性能文件在相同的子目录结构下。
    """
    rel = answer_path.relative_to(answer_base)  # 例如 p4s16t120/scidocs-plaid-top100-...tsv
    # 生成性能文件名（与答案文件名类似，但替换关键字）
    perf_name = answer_path.name
    if "-plaid-top" in perf_name:
        perf_name = perf_name.replace("-plaid-top", "-retrieval-plaid-top", 1).replace(".tsv", ".json")
    elif "-dessert-top" in perf_name:
        perf_name = perf_name.replace("-dessert-top", "-retrieval-dessert-top", 1).replace(".tsv", ".json")
    elif "-MUVERA-top" in perf_name:
        perf_name = perf_name.replace("-MUVERA-top", "-retrieval-MUVERA-top", 1).replace(".tsv", ".json")
    elif "-IGP-top" in perf_name:
        perf_name = perf_name.replace("-IGP-top", "-retrieval-IGP-top", 1).replace(".tsv", ".json")
    else:
        # 通用回退：替换第一个 -top 为 -retrieval-
        perf_name = perf_name.replace("-top", "-retrieval-", 1).replace(".tsv", ".json")
    return performance_base / rel.parent / perf_name


def metric_at_k(retrieved: list[int], relevant: list[int], k: int) -> tuple[float, float, float, float]:
    relevant_set = set(relevant)
    prefix = retrieved[:k]
    hit_positions = [rank for rank, pid in enumerate(prefix, start=1) if pid in relevant_set]

    recall = 1.0 if not relevant else len(set(prefix) & relevant_set) / len(relevant_set)
    success = 1.0 if hit_positions else 0.0
    mrr = 1.0 / hit_positions[0] if hit_positions else 0.0

    dcg = 0.0
    for rank, pid in enumerate(prefix, start=1):
        if pid in relevant_set:
            from math import log2
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = 0.0
    for rank in range(1, ideal_hits + 1):
        from math import log2
        ideal_dcg += 1.0 / log2(rank + 1)
    ndcg = (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    return recall, mrr, success, ndcg


def load_performance(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value):
    if value is None:
        return None
    return float(value)


def choose_answer_files(answer_dir: Path, dataset: str, method: str | None) -> list[Path]:
    pattern = f"{dataset}-*.tsv" if method is None else f"{dataset}-{method}-*.tsv"
    return sorted(answer_dir.glob(pattern))


def write_csv(rows: list[dict], output_csv: Path, k_values: list[int]) -> None:
    base_fields = [
        "answer_file",
        "performance_file",
        "method",
        "n_query",
        "topk",
        "qps",
        "total_query_time_ms",
        "average_query_time_ms",
    ]
    metric_fields = []
    for k in k_values:
        metric_fields.extend([f"recall@{k}", f"mrr@{k}", f"success@{k}", f"ndcg@{k}"])

    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in set(base_fields + metric_fields)
        }
    )
    fieldnames = base_fields + metric_fields + extra_fields

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(
    *,
    answer_path: Path,
    performance_path: Path,
    groundtruth: dict[int, list[int]],
    k_values: Iterable[int],
) -> dict:
    results = parse_answer(answer_path)
    performance = load_performance(performance_path)

    for qid in groundtruth:
        results.setdefault(qid, [])

    search_time = performance["search_time"]
    n_query = int(performance["n_query"])
    total_query_time_ms = float(search_time["total_query_time_ms"])
    qps = (1000.0 * n_query / total_query_time_ms) if total_query_time_ms > 0 else None

    if "-plaid-" in answer_path.name:
        method = "plaid"
    elif "-dessert-" in answer_path.name:
        method = "dessert"
    elif "-MUVERA-" in answer_path.name:
        method = "MUVERA"
    elif "-IGP-" in answer_path.name:
        method = "IGP"
    else:
        method = "unknown"

    row = {
        "answer_file": answer_path.name,
        "performance_file": performance_path.name,
        "method": method,
        "n_query": n_query,
        "topk": int(performance["topk"]),
        "qps": qps,
        "total_query_time_ms": total_query_time_ms,
        "average_query_time_ms": to_float(search_time.get("average_query_time_ms")),
    }

    retrieval_cfg = performance.get("retrieval", {})
    build_cfg = performance.get("build_index", {})
    for key, value in build_cfg.items():
        row[f"build_{key}"] = value
    for key, value in retrieval_cfg.items():
        row[f"retrieval_{key}"] = value

    for k in k_values:
        recall_l = []
        mrr_l = []
        success_l = []
        ndcg_l = []
        for qid in sorted(groundtruth.keys()):
            recall, mrr, success, ndcg = metric_at_k(results.get(qid, []), groundtruth[qid], k)
            recall_l.append(recall)
            mrr_l.append(mrr)
            success_l.append(success)
            ndcg_l.append(ndcg)

        n_eval = max(len(recall_l), 1)
        row[f"recall@{k}"] = sum(recall_l) / n_eval
        row[f"mrr@{k}"] = sum(mrr_l) / n_eval
        row[f"success@{k}"] = sum(success_l) / n_eval
        row[f"ndcg@{k}"] = sum(ndcg_l) / n_eval

    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Plaid/Dessert/MUVERA/IGP outputs against local integer-id ground truth."
    )
    parser.add_argument("--username", type=str, default="ali")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, default=None, choices=[None, "plaid", "dessert", "MUVERA", "IGP"])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 100])

    # new !!!
    parser.add_argument("--index_subdir", type=str, default="",
                        help="Subdirectory under Result/answer/{dataset}/{method}/ (e.g., p4s16t120). If not set, recursively evaluate all subdirs.")
    args = parser.parse_args()

    runtime_root = Path(f"/data1/{args.username}/Dataset/multi-vector-retrieval")
    manifest_path = args.manifest or (runtime_root / "FlatData" / args.dataset / "manifest.json")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    groundtruth_path = Path(manifest["prepared"]["groundtruth_jsonl"])

    #answer_dir = runtime_root / "Result" / "answer"
    #performance_dir = runtime_root / "Result" / "performance"
    #output_csv = args.output_csv or (runtime_root / "Result" / "performance" / f"{args.method}-{args.dataset}-flat-eval-summary.csv")

    # 构建答案和性能的基础目录（含 dataset 和 method）
    answer_base = runtime_root / "Result" / "answer" / args.dataset
    performance_base = runtime_root / "Result" / "performance" / args.dataset
    if args.method:
        answer_base = answer_base / args.method
        performance_base = performance_base / args.method

    # 如果指定了子目录，则精确查找；否则递归所有子目录
    if args.index_subdir:
        answer_dir = answer_base / args.index_subdir
        answer_files = sorted(answer_dir.glob(f"{args.dataset}*.tsv"))
    else:
        answer_files = sorted(answer_base.rglob(f"{args.dataset}*.tsv"))

    if not answer_files:
        raise FileNotFoundError(f"no answer files found under {answer_base} for dataset={args.dataset}, method={args.method}")

    groundtruth = read_groundtruth(groundtruth_path)
    #answer_files = choose_answer_files(answer_dir=answer_dir, dataset=args.dataset, method=args.method)
    #if not answer_files:
    #    raise FileNotFoundError(f"no answer files found in {answer_dir} for dataset={args.dataset}, method={args.method}")

    rows = []
    for answer_path in answer_files:
        #performance_path = infer_performance_path(answer_path, performance_dir)
        performance_path = get_performance_path(answer_path, answer_base, performance_base)
        if not performance_path.exists():
            raise FileNotFoundError(f"missing performance file for {answer_path}: {performance_path}")
        rows.append(
            summarize_run(
                answer_path=answer_path,
                performance_path=performance_path,
                groundtruth=groundtruth,
                k_values=args.k_values,
            )
        )

    # 生成输出 CSV 路径
    if args.output_csv is None:
        # 如果指定了子目录，则 CSV 也放在子目录下
        if args.index_subdir:
            out_dir = performance_base / args.index_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            output_csv = out_dir / f"{args.method}-{args.dataset}-flat-eval-summary.csv"
        else:
            output_csv = runtime_root / "Result" / "performance" / f"{args.method}-{args.dataset}-flat-eval-summary.csv"
    else:
        output_csv = args.output_csv

    write_csv(rows=rows, output_csv=output_csv, k_values=args.k_values)
    print(f"wrote evaluation summary to {output_csv}")


if __name__ == "__main__":
    main()
