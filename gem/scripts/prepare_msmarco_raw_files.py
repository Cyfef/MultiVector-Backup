#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def symlink_force(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)


def write_chunk_counts(offsets_path: Path, output_path: Path) -> np.ndarray:
    offsets = np.load(offsets_path, mmap_mode="r")
    chunk_counts = np.diff(offsets).astype(np.int64, copy=False)
    np.save(output_path, chunk_counts)
    return chunk_counts


def write_dense_qrels(query_ids_path: Path, qrels_source: Path, output_path: Path) -> tuple[int, int]:
    with query_ids_path.open("r", encoding="utf-8") as f:
        query_ids = [int(line.strip()) for line in f if line.strip()]

    qid_to_dense = {qid: idx for idx, qid in enumerate(query_ids)}
    kept = 0
    skipped = 0

    with qrels_source.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        header = src.readline()
        if not header.startswith("query-id"):
            parts = header.rstrip("\n").split("\t")
            if len(parts) >= 2:
                qid = int(parts[0])
                pid = int(parts[1])
                dense_qid = qid_to_dense.get(qid)
                if dense_qid is not None:
                    dst.write(f"{dense_qid}\t{pid}\n")
                    kept += 1
                else:
                    skipped += 1

        for line in src:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            qid = int(parts[0])
            pid = int(parts[1])
            dense_qid = qid_to_dense.get(qid)
            if dense_qid is None:
                skipped += 1
                continue
            dst.write(f"{dense_qid}\t{pid}\n")
            kept += 1

    return kept, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw MSMARCO ColBERT files for GEM preprocessing.")
    parser.add_argument(
        "--raw-source-dir",
        type=Path,
        default=Path("/data1/liuyaoyang/Papers/ACFDE/output/msmarco/colbert"),
        help="Directory containing corpus_points/query_points and their offsets.",
    )
    parser.add_argument(
        "--raw-target-dir",
        type=Path,
        default=Path("/data/ali/msmarco-colbert"),
        help="Directory where the GEM-compatible raw filenames will live.",
    )
    parser.add_argument(
        "--qrels-source",
        type=Path,
        default=Path("/home/ali/gem-baseline/msmarco_evaluation/qrels/dev.tsv"),
        help="Original MSMARCO qrels file with query-id/corpus-id columns.",
    )
    parser.add_argument(
        "--query-ids",
        type=Path,
        default=Path("/home/ali/EMVB/aux_data/msmarco/queries_dev_small_idonly.tsv"),
        help="Dense-order query id list matching query_points.npy.",
    )
    parser.add_argument(
        "--gem-output-root",
        type=Path,
        default=Path("/data/ali/msmarco-gem-data"),
        help="GEM dataset root where qdata/qrels.tsv will be written.",
    )
    parser.add_argument(
        "--dataset-stem",
        type=str,
        default="msmarco-large",
        help="Stem used for the exposed raw filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_source = args.raw_source_dir
    raw_target = args.raw_target_dir
    gem_output_root = args.gem_output_root
    qdata_dir = gem_output_root / "qdata"
    qdata_dir.mkdir(parents=True, exist_ok=True)
    raw_target.mkdir(parents=True, exist_ok=True)

    doc_emb_target = raw_target / f"full_multi_embeddings_{args.dataset_stem}.npy"
    query_emb_target = raw_target / f"full_multi_embeddings_{args.dataset_stem}_query.npy"
    doc_chunks_target = raw_target / f"full_multi_chunk_num_{args.dataset_stem}.npy"
    query_chunks_target = raw_target / f"full_multi_chunk_num_{args.dataset_stem}_query.npy"

    symlink_force(doc_emb_target, raw_source / "corpus_points.npy")
    symlink_force(query_emb_target, raw_source / "query_points.npy")

    doc_chunks = write_chunk_counts(raw_source / "corpus_offsets.npy", doc_chunks_target)
    query_chunks = write_chunk_counts(raw_source / "query_offsets.npy", query_chunks_target)
    kept, skipped = write_dense_qrels(args.query_ids, args.qrels_source, qdata_dir / "qrels.tsv")

    print("Prepared raw files:")
    print(f"  {doc_emb_target}")
    print(f"  {doc_chunks_target} shape={doc_chunks.shape} dtype={doc_chunks.dtype} sum={int(doc_chunks.sum())}")
    print(f"  {query_emb_target}")
    print(f"  {query_chunks_target} shape={query_chunks.shape} dtype={query_chunks.dtype} sum={int(query_chunks.sum())}")
    print(f"  {qdata_dir / 'qrels.tsv'} kept={kept} skipped={skipped}")


if __name__ == "__main__":
    main()
