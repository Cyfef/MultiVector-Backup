import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, "baseline", "ColBERT"))

from baseline.ColBERT import run as colbert_run
from script.flat_multivector.prepare_flat_multivector_dataset import write_plaid_groundtruth_tsvs


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Plaid index for a dataset prepared by prepare_flat_multivector_dataset.py."
    )
    parser.add_argument("--username", type=str, default="ali")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--manifest", type=Path, default=None)

    # new !!!
    parser.add_argument("--num_partitions_override", type=int, default=None)
    parser.add_argument("--num_partitions_multiplier", type=int, default=None)
    parser.add_argument("--kmeans_sample_multiplier", type=int, default=16)
    parser.add_argument("--typical_doclen", type=int, default=120)

    parser.add_argument("--index_subdir", type=str, default="", help="Subdirectory under Index/{dataset} to store this index")
    # new !!!

    args = parser.parse_args()

    runtime_root = Path(f"/data1/{args.username}/Dataset/multi-vector-retrieval")
    manifest_path = args.manifest or (runtime_root / "FlatData" / args.dataset / "manifest.json")
    manifest = load_manifest(manifest_path)

    transformed_embeddings = manifest["prepared"]["doc_transformed_embeddings"]
    groundtruth_jsonl = manifest["prepared"]["groundtruth_jsonl"]
    doc_count_file = manifest["prepared"]["doc_count_file"]
    prepared_query_lens = manifest["prepared"]["prepared_query_lens"]
    query_embeddings = manifest["source"]["query_embeddings"]
    dim = int(manifest["counts"]["dim"])

    query_lens = np.load(prepared_query_lens)
    if int(len(query_lens)) != int(manifest["counts"]["n_query"]):
        raise ValueError("query length file does not match manifest query count")

    colbert_run.build_index_official(
        username=args.username,
        dataset=args.dataset,

        # new !!!
        num_partitions_override=args.num_partitions_override,          # 直接指定中心数
        num_partitions_multiplier=args.num_partitions_multiplier,           
        kmeans_sample_multiplier = args.kmeans_sample_multiplier,
        typical_doclen= args.typical_doclen,
        
        subdir=args.index_subdir,
        # new !!!

        embedding_folder=transformed_embeddings,
        input_query_embedding_file=query_embeddings,
        gt_file=groundtruth_jsonl,
        doc_count_file=doc_count_file,
        datasets_with_embeddings=[args.dataset],
        dataset_dim_mappings={args.dataset: dim},
        query_embedding_len_file=prepared_query_lens,
    )

    write_plaid_groundtruth_tsvs(
        dataset=args.dataset,
        embedding_dir=runtime_root / "Embedding" / args.dataset,
        groundtruth_jsonl=Path(groundtruth_jsonl),
        n_query=int(manifest["counts"]["n_query"]),
    )

    print(f"Plaid build completed for {args.dataset}")


if __name__ == "__main__":
    main()
