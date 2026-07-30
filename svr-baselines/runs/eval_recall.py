#!/usr/bin/env python3
import argparse
import csv
import struct


def read_ivecs(path: str):
    rows = []
    with open(path, "rb") as f:
        while True:
            raw = f.read(4)
            if not raw:
                break
            k = struct.unpack("<i", raw)[0]
            vals = struct.unpack("<" + "i" * k, f.read(4 * k))
            rows.append(list(vals))
    return rows


def read_results(path: str, k: int):
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            qid = int(row[0])
            pid = int(row[1])
            rank = int(row[2])
            if rank > k:
                continue
            out.setdefault(qid, {})[rank] = pid
    ordered = []
    for qid in sorted(out):
        ranks = out[qid]
        ordered.append([ranks[r] for r in sorted(ranks)])
    return ordered


def recall_at_k(gt, pred, k: int) -> float:
    total = min(len(gt), len(pred))
    hits = 0.0
    for i in range(total):
        gt_set = set(gt[i][:k])
        pred_set = set(pred[i][:k])
        if gt_set:
            hits += len(gt_set & pred_set) / len(gt_set)
    return hits / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    gt = read_ivecs(args.groundtruth)
    pred = read_results(args.results, args.k)
    print(f"queries_gt={len(gt)}")
    print(f"queries_pred={len(pred)}")
    print(f"recall@{args.k}={recall_at_k(gt, pred, args.k):.6f}")


if __name__ == "__main__":
    main()
