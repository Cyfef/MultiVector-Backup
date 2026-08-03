# Unified and Efficient Approach for Multi-Vector Similarity Search

## Abstract

Multi-Vector Similarity Search is essential for fine-grained semantic retrieval in many real-world applications, offering richer representations than traditional single-vector paradigms. Due to the lack of native multi-vector index, existing methods rely on a filter-and-refine framework built upon single-vector indexes. By treating token vectors within each multi-vector object in isolation and ignoring their correlations, these methods face an inherent dilemma: aggressive filtering sacrifices recall, while conservative filtering incurs prohibitive computational cost during refinement. To address this limitation, we propose MV-HNSW, the first native hierarchical graph index designed for multi-vector data. MV-HNSW introduces a novel edge-weight function that satisfies essential properties (symmetry, cardinality robustness, and query consistency) for graph-based indexing, an accelerated multi-vector similarity computation algorithm, and an augmented search strategy that dynamically discovers topologically disconnected yet relevant candidates. Extensive experiments on seven real-world datasets show that MV-HNSW achieves state-of-the-art search performance, maintaining over 90% recall while reducing search latency by up to 14.0× compared to existing methods.

## Fullpaper

The full version of our paper can be obtained in `fullpaper.pdf`.

## Dataset

1. Download the opensource datasets.
LoTTE: downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz
Ms Macro: https://huggingface.co/datasets/microsoft/ms_marco?spm=a2ty_o01.29997172.0.0.737b55fbr86b9y

2. Embedding the corpus for multi-vector datasets with ColBERTv2 model with default settings.

## Environment

OS: Ubuntu 24.04.2 LTS

GCC/G++: >= 13.3.0

CMake: >= 3.28.3

Python: >= 3.10.16

## Compile and run our algorithms

1. Compile the source code.

```
mkdir build
cd build
cmake ..
make
```

2. Test the queries of the datasets.