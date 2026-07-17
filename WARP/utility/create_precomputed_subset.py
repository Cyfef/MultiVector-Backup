#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a BEIR-style subset dataset and matching packed ColBERT corpus "
            "embeddings from a larger precomputed dataset."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset-name")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--dataset-root",
        default="/data1/liuyaoyang/Papers/ACFDE/datasets",
    )
    parser.add_argument(
        "--embedding-root",
        default="/data1/liuyaoyang/Papers/ACFDE/output",
    )
    parser.add_argument("--encoder", default="colbert")
    parser.add_argument(
        "--background-docs",
        type=int,
        default=500_000,
        help="Random non-qrel documents to include in addition to all qrel docs.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dataset-root",
        default="/data/ali/acfde-subsets/datasets",
    )
    parser.add_argument(
        "--output-embedding-root",
        default="/data/ali/acfde-subsets/output",
    )
    return parser.parse_args()


def default_subset_name(dataset: str, split: str, background_docs: int, seed: int) -> str:
    if background_docs % 1_000_000 == 0:
        bg_label = f"{background_docs // 1_000_000}m"
    elif background_docs % 1000 == 0:
        bg_label = f"{background_docs // 1000}k"
    else:
        bg_label = str(background_docs)
    return f"{dataset}.{split}.bg={bg_label}.plusqrels.seed={seed}"


def load_forced_doc_ids(qrels_path: Path) -> set[str]:
    forced_doc_ids: set[str] = set()
    with open(qrels_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            forced_doc_ids.add(str(row["corpus-id"]))
    return forced_doc_ids


def reservoir_sample_positions(
    corpus_path: Path,
    forced_doc_ids: set[str],
    background_docs: int,
    seed: int,
) -> tuple[list[int], dict[str, int], int]:
    rng = random.Random(seed)
    sampled_positions: list[int] = []
    forced_positions: dict[str, int] = {}
    non_forced_seen = 0
    corpus_count = 0

    with open(corpus_path, encoding="utf-8") as f:
        for corpus_count, line in enumerate(f, start=1):
            row = json.loads(line)
            doc_id = str(row["_id"])

            if doc_id in forced_doc_ids:
                forced_positions[doc_id] = corpus_count - 1
                continue

            if len(sampled_positions) < background_docs:
                sampled_positions.append(corpus_count - 1)
            else:
                choice = rng.randint(0, non_forced_seen)
                if choice < background_docs:
                    sampled_positions[choice] = corpus_count - 1

            non_forced_seen += 1

    missing = forced_doc_ids.difference(forced_positions)
    if missing:
        raise ValueError(f"Missing {len(missing)} qrel documents from corpus.jsonl")

    return sampled_positions, forced_positions, corpus_count


def write_subset_corpus(
    source_corpus_path: Path,
    selected_positions: np.ndarray,
    output_corpus_path: Path,
) -> None:
    output_corpus_path.parent.mkdir(parents=True, exist_ok=True)

    with open(source_corpus_path, encoding="utf-8") as src, open(
        output_corpus_path, "w", encoding="utf-8"
    ) as dst:
        target_idx = 0
        next_position = int(selected_positions[target_idx])

        for position, line in enumerate(src):
            if position != next_position:
                continue

            dst.write(line)
            target_idx += 1
            if target_idx >= len(selected_positions):
                break
            next_position = int(selected_positions[target_idx])


def copy_queries_and_qrels(
    dataset_dir: Path,
    output_dataset_dir: Path,
) -> None:
    shutil.copy2(dataset_dir / "queries.jsonl", output_dataset_dir / "queries.jsonl")

    qrels_src = dataset_dir / "qrels"
    qrels_dst = output_dataset_dir / "qrels"
    qrels_dst.mkdir(parents=True, exist_ok=True)
    for qrels_file in qrels_src.glob("*.tsv"):
        shutil.copy2(qrels_file, qrels_dst / qrels_file.name)


def create_subset_corpus_embeddings(
    source_embedding_dir: Path,
    output_embedding_dir: Path,
    selected_positions: np.ndarray,
) -> tuple[int, int]:
    output_embedding_dir.mkdir(parents=True, exist_ok=True)

    source_offsets = np.load(source_embedding_dir / "corpus_offsets.npy", mmap_mode="r").astype(
        np.int64, copy=False
    )
    source_points = np.load(source_embedding_dir / "corpus_points.npy", mmap_mode="r")

    selected_starts = source_offsets[selected_positions]
    selected_ends = source_offsets[selected_positions + 1]
    doc_lengths = selected_ends - selected_starts

    subset_offsets = np.zeros(len(selected_positions) + 1, dtype=np.int64)
    subset_offsets[1:] = np.cumsum(doc_lengths)
    total_embeddings = int(subset_offsets[-1])
    dim = int(source_points.shape[1])

    subset_points = np.lib.format.open_memmap(
        output_embedding_dir / "corpus_points.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_embeddings, dim),
    )

    for idx, source_doc_position in enumerate(selected_positions):
        src_start = int(source_offsets[source_doc_position])
        src_end = int(source_offsets[source_doc_position + 1])
        dst_start = int(subset_offsets[idx])
        dst_end = int(subset_offsets[idx + 1])
        subset_points[dst_start:dst_end] = source_points[src_start:src_end]

        if (idx + 1) % 10_000 == 0:
            print(
                f"copied {idx + 1:,} / {len(selected_positions):,} subset docs",
                flush=True,
            )

    np.save(output_embedding_dir / "corpus_offsets.npy", subset_offsets)
    np.save(output_embedding_dir / "selected_doc_positions.npy", selected_positions)

    return len(selected_positions), total_embeddings


def copy_query_embeddings(
    source_embedding_dir: Path,
    output_embedding_dir: Path,
) -> None:
    for name in ["query_points.npy", "query_offsets.npy"]:
        shutil.copy2(source_embedding_dir / name, output_embedding_dir / name)


def write_metadata(
    output_dataset_dir: Path,
    dataset: str,
    subset_name: str,
    split: str,
    seed: int,
    background_docs: int,
    forced_doc_count: int,
    selected_doc_count: int,
    total_corpus_docs: int,
    total_embeddings: int,
) -> None:
    metadata = {
        "source_dataset": dataset,
        "subset_name": subset_name,
        "split": split,
        "seed": seed,
        "background_docs": background_docs,
        "forced_doc_count": forced_doc_count,
        "selected_doc_count": selected_doc_count,
        "total_corpus_docs": total_corpus_docs,
        "total_embeddings": total_embeddings,
    }
    with open(output_dataset_dir / "subset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()

    subset_name = args.subset_name or default_subset_name(
        args.dataset,
        args.split,
        args.background_docs,
        args.seed,
    )

    dataset_dir = Path(args.dataset_root) / args.dataset
    embedding_dir = Path(args.embedding_root) / args.dataset / args.encoder
    output_dataset_dir = Path(args.output_dataset_root) / subset_name
    output_embedding_dir = Path(args.output_embedding_root) / subset_name / args.encoder

    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    output_embedding_dir.mkdir(parents=True, exist_ok=True)

    qrels_path = dataset_dir / "qrels" / f"{args.split}.tsv"
    corpus_path = dataset_dir / "corpus.jsonl"

    forced_doc_ids = load_forced_doc_ids(qrels_path)
    sampled_positions, forced_positions, corpus_count = reservoir_sample_positions(
        corpus_path=corpus_path,
        forced_doc_ids=forced_doc_ids,
        background_docs=args.background_docs,
        seed=args.seed,
    )

    selected_positions = np.array(
        sorted(set(sampled_positions).union(forced_positions.values())),
        dtype=np.int64,
    )

    print(
        f"subset={subset_name} forced_docs={len(forced_positions):,} "
        f"background_docs={len(sampled_positions):,} "
        f"selected_docs={len(selected_positions):,} corpus_docs={corpus_count:,}",
        flush=True,
    )

    write_subset_corpus(
        source_corpus_path=corpus_path,
        selected_positions=selected_positions,
        output_corpus_path=output_dataset_dir / "corpus.jsonl",
    )
    copy_queries_and_qrels(dataset_dir, output_dataset_dir)
    selected_doc_count, total_embeddings = create_subset_corpus_embeddings(
        source_embedding_dir=embedding_dir,
        output_embedding_dir=output_embedding_dir,
        selected_positions=selected_positions,
    )
    copy_query_embeddings(embedding_dir, output_embedding_dir)
    write_metadata(
        output_dataset_dir=output_dataset_dir,
        dataset=args.dataset,
        subset_name=subset_name,
        split=args.split,
        seed=args.seed,
        background_docs=args.background_docs,
        forced_doc_count=len(forced_positions),
        selected_doc_count=selected_doc_count,
        total_corpus_docs=corpus_count,
        total_embeddings=total_embeddings,
    )

    print(f"dataset subset: {output_dataset_dir}")
    print(f"embedding subset: {output_embedding_dir}")


if __name__ == "__main__":
    main()
