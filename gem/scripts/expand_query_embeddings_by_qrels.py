#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand unique-query embeddings into pair-aligned embeddings using qrels rows."
    )
    parser.add_argument("--query-embs", type=Path, required=True)
    parser.add_argument("--query-id-jsonl", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-query-embs", type=Path, required=True)
    parser.add_argument("--output-qrels", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--copy-batch-size", type=int, default=4096)
    return parser.parse_args()


def load_query_ids(path: Path) -> list[str]:
    query_ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            query_ids.append(str(json.loads(line)["_id"]))
    if not query_ids:
        raise ValueError(f"No query ids found in {path}")
    return query_ids


def load_qrels(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    header: list[str] = []
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if line_no == 1 and parts[:2] == ["query-id", "corpus-id"]:
                header.append(line)
                continue
            if len(parts) < 2:
                continue
            rows.append((parts[0], line))
    if not rows:
        raise ValueError(f"No qrels rows found in {path}")
    return header, rows


def main() -> None:
    args = parse_args()
    query_embs = np.load(args.query_embs, mmap_mode="r")
    if query_embs.ndim != 3:
        raise ValueError(f"Expected query embeddings [N, L, D], got {query_embs.shape}")

    query_ids = load_query_ids(args.query_id_jsonl)
    if len(query_ids) != query_embs.shape[0]:
        raise ValueError(
            f"Query id count mismatch: {len(query_ids)} ids vs {query_embs.shape[0]} embedding rows"
        )
    id_to_index = {qid: idx for idx, qid in enumerate(query_ids)}

    header, qrels_rows = load_qrels(args.qrels)
    pair_indices = []
    pair_lines = []
    missing_qids = 0
    for qid, raw_line in qrels_rows:
        idx = id_to_index.get(str(qid))
        if idx is None:
            missing_qids += 1
            continue
        pair_indices.append(idx)
        pair_lines.append(raw_line)
    if not pair_indices:
        raise ValueError("No qrels rows matched the provided query ids.")

    args.output_query_embs.parent.mkdir(parents=True, exist_ok=True)
    args.output_qrels.parent.mkdir(parents=True, exist_ok=True)
    out_shape = (len(pair_indices), query_embs.shape[1], query_embs.shape[2])
    out_embs = np.lib.format.open_memmap(
        args.output_query_embs, mode="w+", dtype=query_embs.dtype, shape=out_shape
    )
    pair_indices_np = np.asarray(pair_indices, dtype=np.int64)
    for start in range(0, len(pair_indices_np), args.copy_batch_size):
        end = min(start + args.copy_batch_size, len(pair_indices_np))
        out_embs[start:end] = np.asarray(query_embs[pair_indices_np[start:end]])
    out_embs.flush()

    with args.output_qrels.open("w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        for line in pair_lines:
            f.write(line + "\n")

    if args.metadata_out is not None:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata_out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "query_embs": str(args.query_embs),
                    "query_id_jsonl": str(args.query_id_jsonl),
                    "qrels": str(args.qrels),
                    "matched_rows": len(pair_indices),
                    "missing_qids": missing_qids,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    print("matched_rows", len(pair_indices))
    print("missing_qids", missing_qids)
    print("output_query_embs", args.output_query_embs)
    print("output_qrels", args.output_qrels)


if __name__ == "__main__":
    main()
