#!/usr/bin/env python3
import argparse
import json
import math
import os
import struct
from typing import Dict, List, Tuple


Qrels = Dict[str, Dict[str, int]]
Results = Dict[str, List[Tuple[str, float]]]


def read_ivecs_qrels(path: str) -> Qrels:
    qrels: Qrels = {}
    with open(path, "rb") as f:
        qid = 0
        while True:
            raw = f.read(4)
            if not raw:
                break
            k = struct.unpack("<i", raw)[0]
            vals = struct.unpack("<" + "i" * k, f.read(4 * k))
            qrels[str(qid)] = {str(docid): 1 for docid in vals}
            qid += 1
    return qrels


def read_text_qrels(path: str) -> Qrels:
    qrels: Qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            elif len(parts) == 3:
                qid, docid, rel = parts[0], parts[1], int(parts[2])
            else:
                raise ValueError(f"unsupported qrels line: {line}")
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def read_qrels(path: str) -> Qrels:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ivecs":
        return read_ivecs_qrels(path)
    return read_text_qrels(path)


def read_results(path: str) -> Results:
    grouped: Dict[str, List[Tuple[int, float, str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            if len(parts) >= 6 and parts[1].upper() == "Q0":
                qid, docid = parts[0], parts[2]
                rank = int(parts[3])
                score = float(parts[4])
            elif len(parts) >= 4:
                qid, docid = parts[0], parts[1]
                rank = int(parts[2])
                score = float(parts[3])
            elif len(parts) == 3:
                qid, docid = parts[0], parts[1]
                rank = int(parts[2])
                score = -float(rank)
            else:
                raise ValueError(f"unsupported results line: {line}")

            grouped.setdefault(qid, []).append((rank, score, docid))

    results: Results = {}
    for qid, rows in grouped.items():
        rows.sort(key=lambda item: (item[0], -item[1], item[2]))
        results[qid] = [(docid, score) for _, score, docid in rows]
    return results


def dcg(rels: List[int]) -> float:
    total = 0.0
    for idx, rel in enumerate(rels, start=1):
        total += (2**rel - 1) / math.log2(idx + 1)
    return total


def metric_at_k(qrel: Dict[str, int], ranked_docs: List[str], k: int) -> Tuple[float, float, float, float]:
    top_docs = ranked_docs[:k]
    rel_count = sum(1 for rel in qrel.values() if rel > 0)
    hits = 0
    ap_sum = 0.0
    graded = []

    for idx, docid in enumerate(top_docs, start=1):
        rel = qrel.get(docid, 0)
        graded.append(rel)
        if rel > 0:
            hits += 1
            ap_sum += hits / idx

    ideal_rels = sorted((rel for rel in qrel.values() if rel > 0), reverse=True)[:k]
    ndcg = dcg(graded) / dcg(ideal_rels) if ideal_rels else 0.0
    map_denom = min(rel_count, k) if k else 0
    mean_ap = ap_sum / map_denom if map_denom else 0.0
    recall = hits / rel_count if rel_count else 0.0
    precision = hits / k if k else 0.0
    return ndcg, mean_ap, recall, precision


def evaluate_standard(qrels: Qrels, results: Results, k_values: List[int], ignore_identical_ids: bool) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    _map = {f"MAP@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    precision = {f"P@{k}": 0.0 for k in k_values}

    for qid, qrel in qrels.items():
        ranked = [docid for docid, _ in results.get(qid, [])]
        if ignore_identical_ids:
            ranked = [docid for docid in ranked if docid != qid]
        for k in k_values:
            q_ndcg, q_map, q_recall, q_precision = metric_at_k(qrel, ranked, k)
            ndcg[f"NDCG@{k}"] += q_ndcg
            _map[f"MAP@{k}"] += q_map
            recall[f"Recall@{k}"] += q_recall
            precision[f"P@{k}"] += q_precision

    denom = len(qrels) if qrels else 1
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / denom, 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / denom, 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / denom, 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / denom, 5)

    return ndcg, _map, recall, precision


def evaluate_mrr(qrels: Qrels, results: Results, k_values: List[int], ignore_identical_ids: bool) -> Dict[str, float]:
    out = {f"MRR@{k}": 0.0 for k in k_values}
    for qid, qrel in qrels.items():
        ranked = [docid for docid, _ in results.get(qid, [])]
        if ignore_identical_ids:
            ranked = [docid for docid in ranked if docid != qid]
        relevant = {docid for docid, rel in qrel.items() if rel > 0}
        for k in k_values:
            for rank, docid in enumerate(ranked[:k], start=1):
                if docid in relevant:
                    out[f"MRR@{k}"] += 1.0 / rank
                    break
    denom = len(qrels) if qrels else 1
    for k in k_values:
        out[f"MRR@{k}"] = round(out[f"MRR@{k}"] / denom, 5)
    return out


def evaluate_recall_cap(qrels: Qrels, results: Results, k_values: List[int], ignore_identical_ids: bool) -> Dict[str, float]:
    out = {f"R_cap@{k}": 0.0 for k in k_values}
    for qid, qrel in qrels.items():
        ranked = [docid for docid, _ in results.get(qid, [])]
        if ignore_identical_ids:
            ranked = [docid for docid in ranked if docid != qid]
        relevant = [docid for docid, rel in qrel.items() if rel > 0]
        for k in k_values:
            denom = min(len(relevant), k)
            if denom == 0:
                continue
            rel_ret = sum(1 for docid in ranked[:k] if qrel.get(docid, 0) > 0)
            out[f"R_cap@{k}"] += rel_ret / denom
    denom = len(qrels) if qrels else 1
    for k in k_values:
        out[f"R_cap@{k}"] = round(out[f"R_cap@{k}"] / denom, 5)
    return out


def evaluate_hole(qrels: Qrels, results: Results, k_values: List[int], ignore_identical_ids: bool) -> Dict[str, float]:
    out = {f"Hole@{k}": 0.0 for k in k_values}
    annotated_corpus = {docid for rels in qrels.values() for docid in rels.keys()}
    for qid in qrels:
        ranked = [docid for docid, _ in results.get(qid, [])]
        if ignore_identical_ids:
            ranked = [docid for docid in ranked if docid != qid]
        for k in k_values:
            if k == 0:
                continue
            hole_docs = sum(1 for docid in ranked[:k] if docid not in annotated_corpus)
            out[f"Hole@{k}"] += hole_docs / k
    denom = len(qrels) if qrels else 1
    for k in k_values:
        out[f"Hole@{k}"] = round(out[f"Hole@{k}"] / denom, 5)
    return out


def evaluate_accuracy(qrels: Qrels, results: Results, k_values: List[int], ignore_identical_ids: bool) -> Dict[str, float]:
    out = {f"Accuracy@{k}": 0.0 for k in k_values}
    for qid, qrel in qrels.items():
        ranked = [docid for docid, _ in results.get(qid, [])]
        if ignore_identical_ids:
            ranked = [docid for docid in ranked if docid != qid]
        relevant = {docid for docid, rel in qrel.items() if rel > 0}
        for k in k_values:
            if any(docid in relevant for docid in ranked[:k]):
                out[f"Accuracy@{k}"] += 1.0
    denom = len(qrels) if qrels else 1
    for k in k_values:
        out[f"Accuracy@{k}"] = round(out[f"Accuracy@{k}"] / denom, 5)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="BEIR-style retrieval evaluator for TSV/TREC results.")
    parser.add_argument("--groundtruth", required=True, help="Qrels path. Supports .ivecs or text qrels.")
    parser.add_argument("--results", required=True, help="Results path. Supports qid-doc-rank TSV, qid-doc-rank-score TSV, or TREC run format.")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 100, 1000],
        help="Cutoffs to evaluate. BEIR defaults to 1 3 5 10 100 1000.",
    )
    parser.add_argument(
        "--ignore-identical-ids",
        action="store_true",
        help="Match BEIR's default behavior of removing retrieved docs whose id matches the query id exactly.",
    )
    parser.add_argument("--output-json", help="Optional path to write the metric summary as JSON.")
    args = parser.parse_args()

    qrels = read_qrels(args.groundtruth)
    results = read_results(args.results)
    k_values = sorted(dict.fromkeys(args.k_values))

    ndcg, _map, recall, precision = evaluate_standard(qrels, results, k_values, args.ignore_identical_ids)
    mrr = evaluate_mrr(qrels, results, k_values, args.ignore_identical_ids)
    recall_cap = evaluate_recall_cap(qrels, results, k_values, args.ignore_identical_ids)
    hole = evaluate_hole(qrels, results, k_values, args.ignore_identical_ids)
    accuracy = evaluate_accuracy(qrels, results, k_values, args.ignore_identical_ids)

    summary = {
        "queries_qrels": len(qrels),
        "queries_results": len(results),
        "k_values": k_values,
        "ignore_identical_ids": args.ignore_identical_ids,
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "recall_cap": recall_cap,
        "hole": hole,
        "accuracy": accuracy,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
