#!/usr/bin/env python3
import argparse
import csv
import mmap
import os
from typing import Dict, List, Tuple


class RawBinMatrix:
    def __init__(self, path: str, dim: int, count: int | None = None) -> None:
        if dim <= 0:
            raise ValueError("--dim must be positive for raw .bin matrices")
        self.path = path
        self.dim = dim
        self.fd = open(path, "rb")
        size = os.path.getsize(path)
        rec_size = dim * 4
        if size % rec_size != 0:
            raise ValueError(f"raw binary size is not divisible by dim for {path}")
        inferred_count = size // rec_size
        if count is None:
            count = inferred_count
        elif count > inferred_count:
            raise ValueError(f"requested {count} vectors from {path}, only {inferred_count} available")
        self.count = count
        self.mm = mmap.mmap(self.fd.fileno(), length=0, access=mmap.ACCESS_READ)
        self.floats = memoryview(self.mm).cast("f")

    def row(self, idx: int):
        if idx < 0 or idx >= self.count:
            raise IndexError(f"row index out of range for {self.path}: {idx}")
        start = idx * self.dim
        end = start + self.dim
        return self.floats[start:end]

    def close(self) -> None:
        self.floats.release()
        self.mm.close()
        self.fd.close()


def read_ranked_results(path: str) -> Dict[int, List[Tuple[int, int]]]:
    grouped: Dict[int, List[Tuple[int, int]]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(f"expected at least 3 TSV columns in {path}: {row}")
            qid = int(row[0])
            docid = int(row[1])
            rank = int(row[2])
            grouped.setdefault(qid, []).append((rank, docid))

    for qid in grouped:
        grouped[qid].sort(key=lambda item: (item[0], item[1]))
    return grouped


def squared_l2(a, b) -> float:
    total = 0.0
    for x, y in zip(a, b):
        diff = float(x) - float(y)
        total += diff * diff
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach -L2^2 scores to rank-only TSV retrieval results.")
    parser.add_argument("--base", required=True, help="Raw float32 base matrix .bin.")
    parser.add_argument("--queries", required=True, help="Raw float32 query matrix .bin.")
    parser.add_argument("--results", required=True, help="Input TSV with qid, docid, rank.")
    parser.add_argument("--output", required=True, help="Output TSV with qid, docid, rank, score.")
    parser.add_argument("--dim", type=int, required=True, help="Vector dimension.")
    parser.add_argument("--num-queries", type=int, help="Optional number of query vectors to expose.")
    args = parser.parse_args()

    base = RawBinMatrix(args.base, args.dim)
    queries = RawBinMatrix(args.queries, args.dim, count=args.num_queries)
    grouped = read_ranked_results(args.results)

    try:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            for qid in sorted(grouped):
                query_vec = queries.row(qid)
                for rank, docid in grouped[qid]:
                    score = -squared_l2(query_vec, base.row(docid))
                    writer.writerow((qid, docid, rank, f"{score:.10f}"))
    finally:
        base.close()
        queries.close()


if __name__ == "__main__":
    main()
