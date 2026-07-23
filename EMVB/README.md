# EMVB

## I. Environment

```bash
conda create -n EMVB python=3.10 -y
conda activate EMVB
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu numpy tqdm cmake make gxx_linux-64 mkl mkl-devel -y
pip install beir
```

For CPU-only indexing:

```bash
conda install -c conda-forge faiss-cpu numpy tqdm cmake make gxx_linux-64 mkl mkl-devel -y
pip install beir
```

The C++ code is compiled with AVX-512 flags in the current `CMakeLists.txt`. Run this to confirm the CPU supports AVX-512:

```bash
lscpu | grep -i avx512
```

If no AVX-512 flags are listed, the binary may fail at runtime with an illegal instruction. In that case, remove the AVX-512-specific compile flags from `CMakeLists.txt` and rebuild, accepting slower search.

Build The C++ Search Binary:

```
cd EMVB
mkdir -p build
cd build
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_CXX_FLAGS="-march=native" ..
make -j"$(nproc)"
```

Expected result: an executable file named `perf_emvb` in `/build` .


## II. Run

1. Build An EMVB Index

```bash
cd EMVB
mkdir -p logs

python -u prepare_emvb_data.py \
  --base-embeddings /data1/chenyifeng/scidocs/colbert/corpus_points.npy \
  --base-offsets /data1/chenyifeng/scidocs/colbert/corpus_offsets.npy \
  --query-embeddings /data1/chenyifeng/scidocs/colbert/query_points.npy \
  --query-offsets /data1/chenyifeng/scidocs/colbert/query_offsets.npy \
  --output-dir /data1/chenyifeng/MultiVector-Backup/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32 \
  --nlist 4096 \
  --pq-m 32 \
  --add-batch-size 200000 \
  --faiss-threads 48 \
  2>&1 | tee /data1/chenyifeng/MultiVector-Backup/EMVB/logs/prepare_scidocs_colbert_simple.log
```

2. Run One Search Manually

This command runs the base `k=10` SciDocs search using one thread.

```bash
cd /data/ali/EMVB
mkdir -p /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
/data1/chenyifeng/MultiVector-Backup/EMVB/build/perf_emvb \
  -k 10 \
  -nprobe 4 \
  -thresh 0.4 \
  -out-second-stage 512 \
  -thresh-query 0.5 \
  -n-doc-to-score 4000 \
  -queries-id-file /data1/chenyifeng/MultiVector-Backup/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/queries_id.txt \
  -alldoclens-path /data1/chenyifeng/MultiVector-Backup/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/alldoclens.npy \
  -index-dir-path /data1/chenyifeng/MultiVector-Backup/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32 \
  -out-file /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/results_k10.tsv \
  > /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/run_k10.log 2>&1
```


3. Evaluate One Search

Evaluate the manual `k=10` SciDocs run:

```bash
cd EMVB

python evaluate_beir_emvb.py \
  --dataset-dir /data1/chenyifeng/scidocs/beir \
  --split test \
  --run /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/results_k10.tsv \
  --k-values 10 \
  --run-log /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/run_k10.log \
  --search-threads 1 \
  --output-json /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/metrics_k10.json \
  --output-csv /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/metrics_k10.csv
```


## III. Notes


1. `MultiVector-Backup/EMVB/prepare_emvb_data.py`


**输入文件（原始数据）**

| 输入来源 / 关联参数 | 文件类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--base-embeddings` | `.npy` 文件 | NumPy 二进制格式 (`np.float32`) | `(total_doc_vectors, embedding_dim)`<br>（所有文档的向量总数 × 向量维度） | **文档的原始嵌入向量**。这是构建索引的原始物料，每一行对应一个文档分块（Chunk）的向量表示。 | **必需** |
| `--doclens` <br>（与 offsets 二选一） | `.npy` 文件 | NumPy 二进制格式 (`np.int64`) | `(num_docs,)`<br>（文档总数） | **每个文档包含的向量数量**。例如，文档 A 有 5 个向量，则对应值为 5。脚本通过累加此数组确定边界。 | **必需**（二选一） |
| `--base-offsets` <br>（与 doclens 二选一） | `.npy` 文件 | NumPy 二进制格式 (`np.int64`) | `(num_docs + 1,)`<br>（文档总数 + 1） | **文档向量的累积偏移量**。以 0 开头，严格递增。脚本会自动计算 `np.diff` 得到等价于 `doclens` 的数组（如 `[0, 5, 12]` 表示文档0有5个向量，文档1有7个）。 | **必需**（二选一） |
| `--query-embeddings` | `.npy` 文件 | NumPy 二进制格式 (`np.float32`) | `(total_query_vectors, embedding_dim)`<br>（所有查询的向量总数 × 向量维度） | **查询的原始嵌入向量**。每一行对应一个查询分块（Token）的向量表示。 | **必需** |
| `--query-doclens` <br>（与 offsets 二选一） | `.npy` 文件 | NumPy 二进制格式 (`np.int64`) | `(num_queries,)`<br>（查询总数） | **每个查询包含的向量数量**（即查询长度）。脚本依据此来将扁平的查询向量矩阵切分成不同的查询。 | **必需**（二选一） |
| `--query-offsets` <br>（与 doclens 二选一） | `.npy` 文件 | NumPy 二进制格式 (`np.int64`) | `(num_queries + 1,)`<br>（查询总数 + 1） | **查询向量的累积偏移量**。功能与 `--base-offsets` 类似，脚本自动将其转为查询长度。 | **必需**（二选一） |

---

**输出文件（供 C++ `perf_emvb` 使用）**

| 文件名 / 路径 | 文件类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `query_embeddings.npy` | `.npy` 文件 | NumPy 二进制格式 (`np.float32`) | `(num_queries, query_max_terms, embedding_dim)` <br>（查询数 × 最大查询长度 × 向量维度） | **填充后的查询张量**。将不等长的原始查询按 `--query-max-terms` 统一长度，不足补 0（`--pad-value`），超长截断。这是 C++ 代码直接读取并逐条送入检索流水线的数据源。 | **必需（核心）** |
| `alldoclens.npy` | `.npy` 文件 | NumPy 二进制格式 (`np.int32`) | `(num_docs,)`<br>（文档总数） | **每个文档的原始分块数量**（32位整型）。C++ 侧用于构建 `emb2pid`（嵌入到文档ID的映射）和校验内存边界。 | **必需** |
| `queries_id.txt` | `.txt` 文件 | 纯文本，每行一个数字 | `(num_queries,)` 行 | **查询 ID 映射表**。从 0 到 N-1 的递增序列。C++ 在输出最终结果时，会读入此文件将内部的 0-based ID 映射为原始查询 ID（本脚本中两者一致）。 | **必需** |
| `residuals.npy` | `.npy` 文件 | NumPy 二进制格式 (`np.uint8`) | `(total_doc_vectors, pq_M)` <br>（文档向量总数 × PQ 子量化器数量） | **PQ 残差编码（Residuals）**。这是每个文档向量经过乘积量化后的压缩码本索引（硬编码为 8-bit）。C++ 在**精排（Phase 4）**阶段需要这些码来计算与查询的近似距离。 | **必需（核心）** |
| `centroids.npy` | `.npy` 文件 | NumPy 二进制格式 (`np.float32`) | `(nlist, embedding_dim)` <br>（聚类中心数 × 向量维度） | **IVF 倒排索引的聚类中心向量**。C++ 在**粗排（Phase 1）**阶段，计算查询向量与这些中心的距离，以决定搜索哪些倒排拉链（即 `nprobe` 的选择）。 | **必需（核心）** |
| `index_assignment.npy` | `.npy` 文件 | NumPy 二进制格式 (`np.uint64`) | `(total_doc_vectors,)`<br>（文档向量总数） | **每个文档向量所属的 IVF 聚类 ID**。用于验证或辅助 C++ 建立向量到聚类中心的映射关系。 | **非必需（辅助）** |
| `pq_centroids.npy` | `.npy` 文件 | NumPy 二进制格式 (`np.float32`) | `(pq_M, 256, sub_dim)` <br>（子量化器数 × 256级 × 子向量维度） | **PQ 的码本（Centroids）**。包含每个子量化器的 256 个中心点。C++ 在解码残差时，利用此码本将压缩码还原为近似浮点向量。 | **必需（核心）** |
| `centroids_to_pids.txt` | `.txt` 文件 | 纯文本，空格分隔的整数 | `(nlist, variable)` <br>（聚类数 × 每行不定长） | **倒排拉链（Inverted Lists）**。**这是最重要的数据结构**。每一行对应一个聚类中心，行内存储属于该聚类的所有**文档 ID（Doc ID）**。C++ 加载后直接用于粗排阶段的文档召回。 | **必需（核心）** |
| `faiss_ivfpq.index` | `.index` 文件 | FAISS 专有序列化格式 | N/A（FAISS 内部结构） | **完整的 FAISS IVFPQ 索引备份**。脚本尝试将训练好的完整索引保存下来，主要用于调试或人工检查 FAISS 原生状态。C++ `perf_emvb` **并不依赖**此文件。 | **非必需（调试用）** |
| `metadata.json` | `.json` 文件 | UTF-8 文本，JSON 格式 | 键值对（K-V） | **元数据记录**。包含本次预处理的所有参数（如 `nlist`, `pq_m`, 运行耗时等），方便后续追溯索引构建环境。 | **非必需（记录用）** |

---

2. `MultiVector-Backup/EMVB/src/perf_emvb.cpp` or `MultiVector-Backup/EMVB/build/perf_emvb`


**输入（命令行参数 + 隐式数据文件）**

| 输入来源 / 关联参数 | 文件类型 / 参数类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `-index-dir-path` | 目录路径 | 文件系统路径 | N/A | 指向 `prepare_emvb_data.py` 生成的索引目录。程序会在此目录下查找多个隐式数据文件。 | **必需** |
| **（隐式）** `query_embeddings.npy` <br>（位于 `-index-dir-path` 下） | `.npy` 文件 | NumPy 二进制 (`np.float32`) | `(num_queries, query_max_terms, embedding_dim)` | **填充后的查询张量**。程序通过 `cnpy` 直接加载，并按行遍历进行检索。 | **必需（核心）** |
| `-alldoclens-path` | `.npy` 文件路径 | NumPy 二进制 (`np.int32`) | `(num_docs,)` | 每个文档包含的原始分块（向量）数量。`DocumentScorer` 用于构建文档边界映射。 | **必需** |
| **（隐式）** `residuals.npy` <br>（位于 `-index-dir-path` 下） | `.npy` 文件 | NumPy 二进制 (`np.uint8`) | `(total_doc_vectors, pq_M)` | 所有文档向量的 PQ 残差编码（硬编码 8-bit）。用于**精排（Phase 4）**阶段的近似距离计算。 | **必需（核心）** |
| **（隐式）** `centroids.npy` <br>（位于 `-index-dir-path` 下） | `.npy` 文件 | NumPy 二进制 (`np.float32`) | `(nlist, embedding_dim)` | IVF 聚类中心向量。用于**粗排（Phase 1）**阶段选择最近的 `nprobe` 个聚类。 | **必需（核心）** |
| **（隐式）** `pq_centroids.npy` <br>（位于 `-index-dir-path` 下） | `.npy` 文件 | NumPy 二进制 (`np.float32`) | `(pq_M, 256, sub_dim)` | PQ 码本（子量化器中心）。用于解码残差码以重构近似向量。 | **必需（核心）** |
| **（隐式）** `centroids_to_pids.txt` <br>（位于 `-index-dir-path` 下） | `.txt` 文件 | 纯文本，空格分隔的整数 | `(nlist, variable)` | **倒排拉链**。每行对应一个聚类，存储属于该聚类的所有**文档 ID（Doc ID）**。用于粗排阶段的文档召回。 | **必需（核心）** |
| `-queries-id-file` | `.txt` 文件 | 纯文本，每行一个整数 | `(num_queries,)` 行 | 查询 ID 映射表。将内部 0-based 索引映射为原始查询 ID，用于输出结果文件。 | **必需** |
| `-k` | 命令行参数 | 整数 | N/A | 返回最近邻的数量（Top-K）。 | **必需** |
| `-nprobe` | 命令行参数 | 整数 | N/A | 粗排阶段要扫描的聚类单元数量（影响召回率与速度）。 | **必需** |
| `-thresh` | 命令行参数 | 浮点数 | N/A | 第一阶段（候选检索）的阈值，用于过滤低质量候选。 | **必需** |
| `-thresh-query` | 命令行参数 | 浮点数 | N/A | 第四阶段（精排打分）的阈值，用于最终筛选文档。 | **必需** |
| `-out-second-stage` | 命令行参数 | 整数 | N/A | 第二阶段过滤后，保留的候选文档数量。 | **必需** |
| `-n-doc-to-score` | 命令行参数 | 整数 | N/A | 第一阶段筛选后，进入后续评分流程的文档数量。 | **必需** |
| `-out-file` | 输出文件路径 | 文件系统路径 | N/A | 指定最终结果的保存路径（此路径本身作为输入参数传入）。 | **必需** |

---

**输出（程序运行结果）**

| 输出目标 | 类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **标准输出（终端）** | 控制台日志 | 纯文本 | N/A（逐行打印） | 打印向量维度、查询数量、每个查询的向量数。搜索结束后会打印 **`Average Elapsed Time per query X`**（单位为纳秒）。 | 实时监控程序运行状态，获取**核心延迟指标（Latency）**，用于性能分析。 |
| **`-out-file` 指定的文件** | 文本文件（TSV） | UTF-8 纯文本，制表符（`\t`）分隔 | `(num_queries × k, 4)` 行<br>（实际行数取决于有效查询数量） | 每行包含 4 列：<br>1. 原始查询 ID（来自 `queries_id.txt`）<br>2. 文档 ID（Doc ID）<br>3. 排名（从 1 到 k）<br>4. 最终得分 | **核心输出**。用于离线评估检索质量，计算召回率（Recall@k）、平均精度（mAP）等指标。 |
| **（隐式）** 无额外二进制文件 | N/A | N/A | N/A | 程序不生成新的索引或向量文件，所有中间结果均在内存中处理。 | N/A |

---


3. `MultiVector-Backup/EMVB/evaluate_beir_emvb.py`


**输入（命令行参数 + 隐式依赖文件）**

| 输入来源 / 关联参数 | 文件类型 / 参数类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--dataset-dir` | 目录路径 | 文件系统路径 | N/A | BEIR 数据集的根目录。脚本会在此目录下自动查找 `corpus.jsonl`、`queries.jsonl` 以及 `qrels/` 子目录。 | **必需** |
| **（隐式）** `corpus.jsonl` <br>（位于 `--dataset-dir` 下） | `.jsonl` 文件 | UTF-8 文本，每行一个 JSON | 每行一个 JSON 对象，包含 `_id` 等字段 | **文档语料库**。脚本按行读取并提取 `_id` 字段，保持文件顺序构建文档 ID 列表（用于将 C++ 的 0-based 索引映射回真实 ID）。 | **必需**（除非用 `--corpus-ids-file` 覆盖） |
| **（隐式）** `queries.jsonl` <br>（位于 `--dataset-dir` 下） | `.jsonl` 文件 | UTF-8 文本，每行一个 JSON | 每行一个 JSON 对象，包含 `_id` 等字段 | **查询集合**。脚本按行读取并提取 `_id` 字段，保持文件顺序构建查询 ID 列表（用于将 C++ 的 0-based 索引映射回真实 ID）。 | **必需**（除非用 `--query-ids-file` 覆盖） |
| **（隐式）** `qrels/{split}.tsv` <br>（位于 `--dataset-dir` 下） | `.tsv` 文件 | UTF-8 纯文本，制表符分隔 | `(num_qrels, 4)` 列：<br>`query-id`, `corpus-id`, `score`, 忽略列 | **标注数据（Ground Truth）**。包含每个查询对应的相关文档 ID 及相关性分数（通常为 0/1 或多级）。`--split` 参数决定使用哪个切分（如 `test.tsv`）。 | **必需** |
| `--run` | `.tsv` / `.txt` 文件路径 | UTF-8 纯文本，制表符分隔 | `(num_results, 4)` 列：<br>`qid`, `doc_index`, `rank`, `score` | **EMVB 检索结果文件**。即 `perf_emvb` 程序通过 `-out-file` 生成的输出文件。包含位置索引形式的查询 ID、文档 ID 及得分。 | **必需** |
| `--query-ids-file` | `.txt` 或 `.jsonl` 文件路径 | 纯文本（每行一个 ID）或 JSONL | `(num_queries,)` 行 | **自定义有序查询 ID 列表**。用于覆盖从 `queries.jsonl` 自动加载的顺序 ID，通常用于处理 `queries_id.txt` 等特殊格式。 | 可选 |
| `--corpus-ids-file` | `.txt` 或 `.jsonl` 文件路径 | 纯文本（每行一个 ID）或 JSONL | `(num_docs,)` 行 | **自定义有序文档 ID 列表**。用于覆盖从 `corpus.jsonl` 自动加载的顺序 ID。 | 可选 |
| `--run-log` | `.txt` / `.log` 文件路径 | UTF-8 纯文本 | 多行文本日志 | **`perf_emvb` 的标准输出日志**。脚本会从中解析 `Average Elapsed Time per query X` 行，提取平均查询耗时（纳秒）。 | 可选 |
| `--avg-query-time-ns` | 命令行参数 | 浮点数 | N/A | **手动指定平均查询耗时（纳秒）**。如果提供，将覆盖从 `--run-log` 解析的时间。 | 可选 |
| `--k-values` | 命令行参数 | 整数列表（空格分隔） | N/A（如 `1 5 10`） | 指定计算 NDCG、Recall、MRR 等指标的截断值（Top-K）。 | 可选（默认 `[1, 3, 5, 10]`） |
| `--query-id-mode` | 命令行参数 | 字符串枚举 | `auto` / `positional` / `direct` | 指定如何解释 run 文件中的查询列：<br>- `positional`：视为 0-based 索引<br>- `direct`：直接视为字符串 ID<br>- `auto`：自动推断 | 可选（默认 `auto`） |
| `--search-threads` | 命令行参数 | 整数 | N/A | 记录检索时使用的线程数（仅用于记录到输出元数据中，不影响计算）。 | 可选（默认 `1`） |
| `--restrict-to-run-queries` | 命令行参数 | 布尔标志（无需值） | N/A | 若开启，只评估 run 文件中实际存在的查询（而非全部 `queries.jsonl` 中的查询）。 | 可选（默认关闭） |

---

**输出（程序运行结果与报告）**

| 输出目标 | 类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **标准输出（终端）** | 控制台日志 | 缩进 JSON 文本 | 嵌套键值对（字典） | 打印完整的评估摘要，包含：<br>- 元数据（数据集路径、查询/文档数量等）<br>- **NDCG@k**、**MAP**、**MRR@k**、**Recall@k**、**Precision@k**<br>- 性能指标（`avg_query_time_ns`、`QPS`） | 实时查看评估结果，快速确认检索质量与速度。 |
| `--output-json` 指定路径 | JSON 文件 | UTF-8 文本，JSON 格式 | 与终端输出结构一致（嵌套字典） | **完整评估报告**。保留所有维度的指标（NDCG、MAP、MRR、Recall、Precision 各自按 K 值组织的子字典）及全部元数据。 | 长期存档、自动化实验管理、多组超参数对比分析。 |
| `--output-csv` 指定路径 | CSV 文件 | UTF-8 文本，逗号分隔 | **单行**，扁平化键值对（将嵌套指标展开为独立列） | 将多维指标（如 `NDCG@10`, `Recall@100`, `QPS`）压缩为一行。 | 批量实验汇总，方便导入 Excel 或 pandas 进行横向对比和趋势分析。 |
| **（隐式）** 无额外文件 | N/A | N/A | N/A | 脚本本身不修改任何输入文件，也不生成新的检索结果或索引文件。 | N/A |


## IV. Evaluation Metrics


| 指标名称 | 英文缩写 / 代码键名 | 核心计算公式 / 计算方法 | 在代码中的来源 | 评价维度 |
| :--- | :--- | :--- | :--- | :--- |
| **归一化折损累计增益** | NDCG@k<br>（如 `ndcg@10`） | **NDCG = DCG / IDCG**<br><br>- **DCG**（折损累计增益）：`DCG@k = Σ_{i=1}^{k} (rel_i / log₂(i+1))`，其中 `rel_i` 是第 i 位文档的相关性分数（BEIR 通常为 0/1 或分级）。<br>- **IDCG**（理想 DCG）：按相关性降序排列前 k 个文档计算出的最大 DCG。<br><br>**计算方式**：每个查询分别计算 NDCG@k，然后对所有查询取**算术平均值**（宏观平均）。 | `EvaluateRetrieval.evaluate()` 返回的 `ndcg` 字典 | **排序质量**（衡量排名位置的准确性，对靠前位置的错误惩罚更重） |
| **平均精度均值** | MAP@k<br>（代码中键名为 `_map`） | **MAP = (1 / |Q|) × Σ_{q=1}^{|Q|} AP@k(q)** <br><br>- **AP@k**（平均精度）：`AP@k = (1 / R_total) × Σ_{i=1}^{k} (P@i × rel_i)`，其中 `P@i` 是前 i 个结果的精确率，`rel_i` 指示第 i 位是否相关（1或0），`R_total` 是该查询总相关文档数。<br><br>**简化理解**：在所有相关文档出现的位置上，计算该位置的精确率并求平均。 | `EvaluateRetrieval.evaluate()` 返回的 `_map` 字典 | **整体检索精度**（综合衡量召回位置和排序质量，对多级相关性敏感） |
| **召回率** | Recall@k<br>（如 `recall@10`） | **Recall@k = (前 k 个结果中相关文档数) / (该查询总相关文档数)** <br><br>如果总相关文档数为 0，则该查询的 Recall 视为 1（避免分母为 0）或 0（视具体实现）。<br><br>**计算方式**：每个查询分别计算 Recall@k，然后对所有查询取**算术平均值**。 | `EvaluateRetrieval.evaluate()` 返回的 `recall` 字典 | **查全率**（衡量系统找到所有相关文档的能力） |
| **精确率** | Precision@k<br>（如 `precision@10`） | **Precision@k = (前 k 个结果中相关文档数) / k** <br><br>直接统计 Top-K 结果中有多少是相关的。<br><br>**计算方式**：每个查询分别计算 Precision@k，然后对所有查询取**算术平均值**。 | `EvaluateRetrieval.evaluate()` 返回的 `precision` 字典 | **查准率**（衡量系统返回的结果中有多少是用户真正想要的） |
| **平均倒数排名** | MRR@k<br>（如 `mrr@10`） | **MRR = (1 / |Q|) × Σ_{q=1}^{|Q|} (1 / rank_first_relevant_q)** <br><br>- `rank_first_relevant_q` 是第一个相关文档在结果列表中的排名（从 1 开始）。<br>- 如果前 k 个结果中没有相关文档，则贡献为 0（若 rank > k 则视为 0）。<br><br>**计算方式**：对每个查询取首个相关文档排名的倒数，然后对所有查询取平均。 | `EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="mrr")` 返回的 `mrr` 字典 | **首位命中质量**（极度关注用户最关心的第一个结果是否正确） |
| **每秒查询数** | QPS<br>（代码键名 `qps`） | **QPS = 1 / avg_query_time_s** <br><br>其中 `avg_query_time_s = avg_query_time_ns / 1_000_000_000.0`（将纳秒转换为秒）。<br><br>如果平均耗时 ≤ 0，则 QPS = 0。 | 由 `resolve_timing()` 函数计算，来源于 `--run-log` 解析或 `--avg-query-time-ns` 手动传入。 | **吞吐量 / 速度**（衡量系统在单位时间内能处理的查询数量，反映工程性能） |
| **平均查询延迟** | avg_query_time_ns / avg_query_time_s | 直接来自 `perf_emvb` 输出的 `Average Elapsed Time per query`（单位：纳秒），或通过参数手动指定。 | `resolve_timing()` 函数解析日志或直接取值。 | **端到端延迟**（衡量单次查询的平均耗时，反映检索响应速度） |

---
