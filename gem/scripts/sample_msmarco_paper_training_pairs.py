#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample MSMARCO training pairs for paper-style GEM shortcut injection."
    )
    parser.add_argument("--train-query-embs", type=Path, required=True)
    parser.add_argument("--train-qrels", type=Path, required=True)
    parser.add_argument("--sample-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-query-embs", type=Path, required=True)
    parser.add_argument("--output-qrels", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--copy-batch-size", type=int, default=4096)
    return parser.parse_args()


def load_qrels_rows(path: Path) -> tuple[list[str], list[str]]:
    header: list[str] = []
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line_no == 1 and line.startswith("query-id"):
                header.append(line)
                continue
            rows.append(line)
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return header, rows


def main() -> None:
    args = parse_args()
    query_embs = np.load(args.train_query_embs, mmap_mode="r")
    if query_embs.ndim != 3:
        raise ValueError(f"Expected train query embeddings [N, L, D], got {query_embs.shape}")

    header, rows = load_qrels_rows(args.train_qrels)
    if len(rows) != query_embs.shape[0]:
        raise ValueError(
            f"Pair-aligned query/qrels mismatch: {query_embs.shape[0]} embeddings vs {len(rows)} qrels rows"
        )

    total_pairs = len(rows)
    sample_count = int(round(total_pairs * args.sample_ratio))
    sample_count = max(1, min(total_pairs, sample_count))
    rng = np.random.default_rng(args.seed)
    sampled_indices = np.sort(rng.choice(total_pairs, size=sample_count, replace=False))

    args.output_query_embs.parent.mkdir(parents=True, exist_ok=True)
    args.output_qrels.parent.mkdir(parents=True, exist_ok=True)

    sampled_shape = (sample_count, query_embs.shape[1], query_embs.shape[2])
    sampled_embs = np.lib.format.open_memmap(
        args.output_query_embs, mode="w+", dtype=query_embs.dtype, shape=sampled_shape
    )
    for out_start in range(0, sample_count, args.copy_batch_size):
        out_end = min(out_start + args.copy_batch_size, sample_count)
        batch_indices = sampled_indices[out_start:out_end]
        sampled_embs[out_start:out_end] = np.asarray(query_embs[batch_indices])
    sampled_embs.flush()

    with args.output_qrels.open("w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        for idx in sampled_indices:
            f.write(rows[int(idx)] + "\n")

    if args.metadata_out is not None:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata_out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_query_embs": str(args.train_query_embs),
                    "train_qrels": str(args.train_qrels),
                    "sample_ratio": args.sample_ratio,
                    "seed": args.seed,
                    "total_pairs": total_pairs,
                    "sampled_pairs": sample_count,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    print("sampled_pairs", sample_count)
    print("total_pairs", total_pairs)
    print("output_query_embs", args.output_query_embs)
    print("output_qrels", args.output_qrels)


if __name__ == "__main__":
    main()
