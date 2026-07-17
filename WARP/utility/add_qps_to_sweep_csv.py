#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill a qps column into an existing sweep CSV."
    )
    parser.add_argument("csv_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "qps" not in fieldnames:
        try:
            insert_at = fieldnames.index("elapsed_sec") + 1
        except ValueError as exc:
            raise ValueError(f"{csv_path} is missing elapsed_sec column") from exc
        fieldnames.insert(insert_at, "qps")

    for row in rows:
        num_queries = float(row["num_queries"])
        elapsed_sec = float(row["elapsed_sec"])
        row["qps"] = f"{(num_queries / elapsed_sec):.3f}"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {csv_path} with qps")


if __name__ == "__main__":
    main()
