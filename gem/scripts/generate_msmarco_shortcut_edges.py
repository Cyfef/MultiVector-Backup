#!/usr/bin/env python3
import argparse
import itertools
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate doc-doc shortcut edges from sampled MSMARCO training qrels."
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        required=True,
        help="Path to MSMARCO train qrels.tsv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the sampled shortcut edge file.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.2,
        help="Probability of keeping each training qrel line before grouping by query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.sample_ratio <= 1.0:
        raise ValueError("--sample-ratio must be in (0, 1].")

    rng = random.Random(args.seed)
    skipped_header = 0
    positives_by_qid: dict[str, list[int]] = defaultdict(list)

    with args.qrels.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if line_no == 1 and parts[:2] == ["query-id", "corpus-id"]:
                skipped_header += 1
                continue
            if len(parts) < 2:
                continue
            qid = parts[0]
            pid = int(parts[1])
            positives_by_qid[qid].append(pid)

    qids = sorted(positives_by_qid)
    sampled_qids = [qid for qid in qids if rng.random() <= args.sample_ratio]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    edge_count = 0
    contributing_queries = 0
    with args.output.open("w", encoding="utf-8") as out:
        for qid in sampled_qids:
            unique_pids = sorted(set(positives_by_qid[qid]))
            if len(unique_pids) < 2:
                continue
            contributing_queries += 1
            for pid1, pid2 in itertools.combinations(unique_pids, 2):
                out.write(f"{pid1}\t{pid2}\n")
                edge_count += 1

    print(f"qrels={args.qrels}")
    print(f"output={args.output}")
    print(f"sample_ratio={args.sample_ratio}")
    print(f"seed={args.seed}")
    print(f"total_queries={len(qids)}")
    print(f"sampled_queries={len(sampled_qids)}")
    print(f"contributing_queries={contributing_queries}")
    print(f"edge_pairs={edge_count}")
    print(f"skipped_header={skipped_header}")


if __name__ == "__main__":
    main()
