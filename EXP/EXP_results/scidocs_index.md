# SciDocs Index

## 1. 当前状态

| 方法 | Build 参数覆盖范围 |
|------|-------------------|
| **Plaid** | `num_partitions_multiplier` ∈ {4, 16, 32}，`typical_doclen` ∈ {10, 120, 500}，`kmeans_sample_multiplier` ∈ {2, 16, 32} |
| **Dessert** | `n_table` ∈ {16, 64, 128} |
| **EMVB** | `nlist` ∈ {2048, 4096, 8192}，`pq_m` ∈ {16, 32, 64} |

---

## 2. Baseline-scale resource snapshot

| 方法 | Baseline build 配置 | build_time (s) | Peak build memory (MB) | index size (MB) | Recall@100 | Search QPS |
|------|----------------------|---------------|------------------------|----------------------------|------------|------------|
| **Plaid** | `num_partitions_multiplier=16, typical_doclen=120, kmeans_sample_multiplier=16` | 161.58 | 679.47 | 184.11 | 0.3430 | 9.94 |
| **Dessert** | `n_table=64` | 240.36 | 1931.35 | 729.68 | 0.3482 | 24.64 |
| **EMVB** | `nlist=4096, pq_m=32` | 112.94 | 3277.60 | 421.27 | 0.3423 | 2.31 |

---

## 3. Plaid

### 3.1 `num_partitions_multiplier`（固定 `typical_doclen=120`，`kmeans_sample_multiplier=16`）

| `num_partitions_multiplier` | build_time (s) | Index size (MB) | Peak build memory (MB) | Recall@100 | QPS |
|-----------------------------|----------------|-----------------|---------------|------------|------|
| 4 | 84.60 | 177.67 | 678.58 | 0.3210 | 13.78 |
| **16 (baseline)** | **161.58** | **184.11** | **679.47** | **0.3430** | **9.94** |
| 32 | 232.04 | 192.25 | 680.00 | 0.3489 | 7.02 |

`num_partitions_multiplier` 从 4 增至 32，build time 增加约 2.7 倍，index size 仅小幅增加（+8%），Recall 从 0.321 提升至 0.349，QPS 从 13.78 降至 7.02。

---

### 3.2 `typical_doclen`（固定 `num_partitions_multiplier=16`，`kmeans_sample_multiplier=16`）

| `typical_doclen` | build_time (s) | Index size (MB) | Peak build memory (MB) | Recall@100 | QPS |
|------------------|----------------|-----------------|---------------|------------|------|
| 10 | 74.77 | 184.53 | 678.00 | 0.3418 | 8.72 |
| **120 (baseline)** | **161.58** | **184.11** | **679.47** | **0.3430** | **9.94** |
| 500 | 148.01 | 184.11 | 676.28 | 0.3430 | 14.04 |

 `typical_doclen` 在 10~500 范围内对 Recall 和 index size 几乎无影响（Recall 稳定在 0.342~0.343），但 `doclen=500` 时 QPS 显著提升（14.04），且 build time 略低于 baseline。

---

### 3.3 `kmeans_sample_multiplier`（固定 `num_partitions_multiplier=16`，`typical_doclen=120`）

| `kmeans_sample_multiplier` | build_time (s) | Index size (MB) | Peak build memory (MB) | Recall@100 | QPS |
|----------------------------|----------------|-----------------|---------------|------------|------|
| 2 | 54.16 | 184.86 | 676.17 | 0.3413 | 8.66 |
| **16 (baseline)** | **161.58** | **184.11** | **679.47** | **0.3430** | **9.94** |
| 32 | 134.25 | 184.11 | 683.49 | 0.3430 | 9.29 |

增大 `kmeans_sample_multiplier` 对 Recall 和 index size 几乎无影响（0.3413→0.3430），但 build time 先升后降。

---

## 4. Dessert 

### `n_table`

| `n_table` | build_time (s) | Index size (MB) | Peak build memory (MB) | Recall@100 | QPS |
|-----------|----------------|-----------------|---------------|------------|------|
| 16 | 238.02 | 209.48 | 1500.67 | 0.3000 | 40.64 |
| **64 (baseline)** | **240.36** | **729.68** | **1931.35** | **0.3482** | **24.64** |
| 128 | 240.63 | 1423.52 | 2578.07 | 0.3556 | 12.45 |

`n_table` 增大几乎不增加 build time（~240s 稳定），但 index size 和 peak mem 分别增长约 6.8 倍和 1.7 倍。Recall 从 0.300 提升至 0.356，QPS 从 40.64 降至 12.45。

---

## 5. EMVB 

### 5.1 `nlist`（固定 `pq_m=32`）

| `nlist` | build_time (s) | Index size (MB) | Peak build memory (MB) | Recall@100 | QPS |
|---------|----------------|-----------------|---------------|------------|------|
| 2048 | 71.91 | 419.22 | 3164.50 | 0.3211 | 2.48 |
| **4096 (baseline)** | **112.94** | **421.27** | **3277.60** | **0.3423** | **2.31** |
| 8192 | 242.72 | 425.28 | 3514.20 | 0.3527 | 3.11 |

 `nlist` 从 2048 增至 8192，build time 增加 3.4 倍，但 index size 仅微增（+1.4%），Recall 提升明显（0.321→0.353），QPS 在 8192 时反而回升至 3.11。

---

### 5.2 `pq_m`（固定 `nlist=4096`）

| `pq_m` | build_time (s) | Index size (MB) | Peak build memory (MB) | Recall@100 | QPS |
|--------|----------------|-----------------|---------------|------------|------|
| 16 | 104.12 | 274.13 | 3030.80 | 0.3340 | 4.84 |
| **32 (baseline)** | **112.94** | **421.27** | **3277.60** | **0.3423** | **2.31** |
| 64 | 132.69 | 715.56 | 3756.70 | 0.3430 | 1.72 |

`pq_m` 从 16 增至 64，index size 暴涨 2.6 倍（274→716 MB），peak mem 增加约 724 MB，但 Recall 仅从 0.334 微升至 0.343，且 QPS 从 4.84 降至 1.72。

---
