# WARP

## I. Environment

```bash
conda env create -f conda_env_cpu.yml
conda activate warp

conda create -n WARP python=3.8 -y
conda activate WARP

pip install -r requirements.txt

conda install "mkl<2025" packaging -y
conda install -c conda-forge gcc_linux-64 gxx_linux-64
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge libxcrypt libxcrypt-dev

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

export CUDAHOSTCXX=$CXX

export PYTHON_BIN="$(which python)"
export PYTHONPATH=/data/ali/WARP:${PYTHONPATH:-}
export TORCH_EXTENSIONS_DIR=/data1/chenyifeng/tmp/torch-ext
export HF_HOME=/data1/chenyifeng/tmp/hf-cache
export TRANSFORMERS_CACHE=/data1/chenyifeng/tmp/hf-cache
```

For a GPU build environment, use `conda_env.yml` instead of `conda_env_cpu.yml`.


## II. Run

1. Build a WARP Index

```bash
cd WARP

export DATASET=scidocs
export EMBEDDING_DIR=/data1/chenyifeng/MultiVector-Backup/WARP/embeddings/colbert
export INDEX_ROOT=/data1/chenyifeng/MultiVector-Backup/WARP/indexes
export INDEX_NAME=beir-scidocs.split=test.precomputed=colbert.nbits=2.rebuild
export CXX=/usr/bin/g++-9   

python utility/index_from_embeddings.py \
  --dataset "$DATASET" \
  --embedding-dir "$EMBEDDING_DIR" \
  --index-root "$INDEX_ROOT" \
  --index-name "$INDEX_NAME" \
  --nbits 2 \
  --threads 48 \
  --max-partitions 32768 \
  --sample-per-centroid 64 \
  --max-sample-embeddings 2000000 \
  --chunk-size 50000
```

Important indexing parameters:

```text
--nbits 2                    residual quantization bits used by this baseline
--max-partitions 32768       upper bound on k-means centroids
--sample-per-centroid 64     training sample budget per centroid
--max-sample-embeddings 2000000
--chunk-size 50000           documents per saved index chunk
```

Expected output files inside `$INDEX_ROOT/$INDEX_NAME/`:

```text
metadata.json
plan.json
centroids.pt
avg_residual.pt
buckets.pt
ivf.pid.pt
0.codes.pt
0.residuals.pt
0.metadata.json
doclens.0.json
```

For SciDocs simple ColBERT, the built index is about `193M`.


2. Run WARP Search

Use `utility/sweep_ncells.py` to run search and evaluate metrics. This command runs one setting and writes both:

  - a metrics CSV, and
  - a ranked TSV run file with `query-id`, `corpus-id`, `rank`, `score`.

```bash
cd WARP

export CPATH=/usr/include:$CPATH
export CFLAGS="-I/usr/include"
export CXXFLAGS="-I/usr/include"

export DATASET=scidocs
export DATASET_DIR=/data1/chenyifeng/MultiVector-Backup/WARP/data/scidocs/beir
export EMBEDDING_DIR=/data1/chenyifeng/MultiVector-Backup/WARP/embeddings/colbert
export INDEX_ROOT=/data1/chenyifeng/MultiVector-Backup/WARP/indexes
export INDEX_NAME=beir-scidocs.split=test.precomputed=colbert.nbits=2.rebuild
export OUT_DIR=/data1/chenyifeng/MultiVector-Backup/WARP/my_runs/scidocs-colbert
mkdir -p "$OUT_DIR"

python utility/sweep_ncells.py \
  --dataset "$DATASET" \
  --dataset-dir "$DATASET_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --index-root "$INDEX_ROOT" \
  --index "$INDEX_NAME" \
  --split test \
  --nbits 2 \
  --k 100 \
  --baseline warp_precomputed_colbert_nbits2_thr0.45_ndocs1024 \
  --centroid-score-threshold 0.45 \
  --ndocs 1024 \
  --ncells 1 \
  --output-csv "$OUT_DIR/scidocs_colbert_verify.csv" \
  --tsv-output "$OUT_DIR/scidocs_colbert_verify.tsv"
```

Expected terminal line:

```text
ncells=1 NDCG@10=0.15076 NDCG@100=0.21951 Recall@100=0.36043
```


## III. Notes

1. `MultiVector-Backup/WARP/utility/index_from_embeddings.py`


**输入（命令行参数 + 隐式依赖文件）**

| 输入来源 / 关联参数 | 文件类型 / 参数类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--dataset` | 字符串参数 | 命令行字符串 | N/A | 数据集名称（如 `msmarco`、`beir-*`），用于生成默认索引名及推断 `split`。 | **必需** |
| `--encoder` | 字符串参数 | 命令行字符串 | N/A | 编码器名称（如 `colbert`），用于生成默认索引名。 | 可选（默认 `colbert`） |
| `--embedding-root` 或 `--embedding-dir` | 目录路径参数 | 文件系统路径 | N/A | 预计算嵌入文件的根目录或直接目录。若使用 `--embedding-root`，则拼接为 `root/dataset/encoder/`；若使用 `--embedding-dir` 则直接使用该路径。 | **必需**（二选一） |
| **（隐式）** `corpus_points.npy` <br>（位于上述目录） | `.npy` 文件 | NumPy 二进制格式 | `(num_embeddings, dim)`，`float32` | 所有文档 token 级嵌入（已展平，按文档顺序拼接）。 | **必需** |
| **（隐式）** `corpus_offsets.npy` | `.npy` 文件 | NumPy 二进制格式 | `(num_docs + 1,)`，`int64` | 每个文档在 `corpus_points.npy` 中的起始和结束索引，用于切分文档边界。 | **必需** |
| **（隐式）** `query_offsets.npy` | `.npy` 文件 | NumPy 二进制格式 | `(num_queries + 1,)`，`int64` | 查询 token 边界（仅用于推断 `query_maxlen`，若不提供则使用默认值 32）。 | 可选 |
| `--checkpoint` | 字符串参数 | 文件系统路径 | N/A | 模型检查点路径，仅作为元数据存入索引，不实际加载。 | 可选（默认固定路径） |
| `--index-root` | 目录路径参数 | 文件系统路径 | N/A | 索引存储的根目录。最终索引将位于 `index_root/index_name` 下。 | **必需**（或通过环境变量 `INDEX_ROOT` 设置） |
| `--index-name` | 字符串参数 | 命令行字符串 | N/A | 自定义索引名称。若不指定，则根据 `dataset`、`encoder`、`split`、`nbits` 自动生成。 | 可选 |
| `--nbits` | 整数参数（1/2/4/8） | 命令行整数 | N/A | 残差量化每维使用的比特数，影响压缩率和检索精度。 | 可选（默认 `2`） |
| `--kmeans-iters` | 整数参数 | 命令行整数 | N/A | K-means 聚类迭代次数。 | 可选（默认 `4`） |
| `--chunk-size` | 整数参数 | 命令行整数 | N/A | 每个索引分块包含的文档数，用于分批次压缩以减少内存峰值。 | 可选（默认 `50_000`） |
| `--max-partitions` | 整数参数 | 命令行整数 | N/A | IVF 聚类中心数的上限。实际中心数会基于嵌入总数自动计算并受此限制。 | 可选（默认 `65_536`） |
| `--sample-per-centroid` | 整数参数 | 命令行整数 | N/A | 每个聚类中心用于训练的采样样本数。 | 可选（默认 `32`） |
| `--max-sample-embeddings` | 整数参数 | 命令行整数 | N/A | 训练 K-means 时可使用的最大样本数（硬上限）。 | 可选（默认 `4_000_000`） |
| `--query-maxlen` | 整数参数 | 命令行整数 | N/A | 查询最大 token 长度，存于索引元数据。若不指定，则从 `query_offsets.npy` 自动推断。 | 可选 |
| `--doc-maxlen` | 整数参数 | 命令行整数 | N/A | 文档最大 token 长度，存于索引元数据。若不指定，则使用 `corpus_offsets` 中最大跨度。 | 可选 |
| `--seed` | 整数参数 | 命令行整数 | N/A | 随机种子，用于采样和 K-means 初始化。 | 可选（默认 `123`） |
| `--threads` | 整数参数 | 命令行整数 | N/A | CPU 线程数，用于 FAISS 和 PyTorch 操作。 | 可选 |
| `--faiss-gpu-python` | 字符串参数 | 文件系统路径（Python 解释器） | N/A | 支持 `faiss-gpu` 的 Python 可执行文件路径。若提供，则调用外部 GPU 辅助脚本加速 K-means 和压缩。 | 可选 |
| `--faiss-gpu-visible-device` | 字符串参数 | 环境变量值（如 `"0"`） | N/A | 指定 GPU 设备 ID（`CUDA_VISIBLE_DEVICES`）。 | 可选 |
| `--faiss-gpu-batch-size` | 整数参数 | 命令行整数 | N/A | 调用 GPU 辅助脚本时，每批处理的嵌入数。 | 可选（默认 `250_000`） |
| `--tmp-root` | 目录路径参数 | 文件系统路径 | N/A | 临时文件存储根目录，用于存放 GPU 辅助脚本的中间文件。 | 可选（默认 `/tmp`） |
| **（隐式）** `faiss_gpu_helper.py` | Python 脚本 | 位于 `REPO_ROOT/utility/` | N/A | 辅助脚本，用于执行 K-means 训练和嵌入压缩。仅在启用 `--faiss-gpu-python` 时被调用。 | 条件必需 |

---

**输出（程序运行结果与生成文件）**

| 输出目标 | 类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **索引根目录**（`--index-root/index_name`） | 目录 | 文件系统目录 | N/A | 包含完整 WARP/ColBERT 索引的所有文件，结构如下： | 供检索系统（如 `warp.search`）直接加载。 |
| └─ `plan.json` | 元数据文件 | UTF-8 JSON | 嵌套字典 | 记录索引构建时的预估参数（分块数、分区数、总嵌入数、平均文档长度等）。 | 构建过程中的校验参考。 |
| └─ `metadata.json` | 元数据文件 | UTF-8 JSON | 嵌套字典 | 最终索引的全局元数据（配置、分块数、分区数、嵌入总数、平均文档长度）。 | 检索时读取配置和统计信息。 |
| └─ `codec/` 或 `codec.*` 文件 | 编码器文件 | NumPy `.npy` 或 PyTorch `.pt` | 聚类中心、量化阈值等 | 残差编码器（`ResidualCodec`）的所有参数，包括 `centroids`、`avg_residual`、`bucket_cutoffs`、`bucket_weights`。 | 用于压缩新文档或解压索引。 |
| └─ `{chunk_id}.codes.pt` / `.npy` | 分块码文件 | PyTorch 或 NumPy 格式 | `(num_embeddings_in_chunk,)`（整型码） | 每个分块中文档 token 的量化码（中心索引）和残差信息（若使用残差量化）。 | 存储压缩后的文档表示，检索时按分块加载。 |
| └─ `{chunk_id}.metadata.json` | 分块元数据 | UTF-8 JSON | 包含文档起始位置、文档长度、嵌入偏移等 | 记录该分块的文档范围及每个文档的 token 数量。 | 辅助定位文档边界和构建全局偏移。 |
| └─ `ivf.pt` / `ivf.*` | IVF 倒排文件 | PyTorch 张量或二进制 | 排序后的文档 ID 列表及每个聚类中心的起始偏移 | 由 `optimize_ivf` 生成的倒排索引结构，用于快速检索。 | 检索时通过聚类中心快速召回候选文档。 |
| **标准输出（控制台）** | 日志信息 | 纯文本（带时间戳） | 多行字符串 | 构建过程的实时进度信息（加载、采样、训练、压缩、最终化等各阶段耗时及状态）。 | 监控构建进度、调试错误。 |
| **临时目录**（`--tmp-root` 下） | 临时文件 | 仅在构建期间存在 | N/A | 若使用 GPU 辅助，会生成 `.npy` 中间文件（如 `kmeans_sample.npy`、`chunk_X.codes.npy`），脚本运行结束后自动清理。 | 用于 GPU 辅助进程间数据传递。 |

---

2. `MultiVector-Backup/WARP/utility/sweep_ncells.py`

**输入（命令行参数 + 隐式依赖文件）**

| 输入来源 / 关联参数 | 文件类型 / 参数类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--dataset` | 字符串参数 | 命令行字符串 | N/A | 数据集名称（如 `msmarco`），用于自动推断 `split` 和生成默认文件名。 | **必需** |
| `--dataset-dir` 或 `--dataset-root` | 目录路径参数 | 文件系统路径 | N/A | BEIR 格式数据集的根目录或直接路径。脚本会在该目录下查找 `corpus.jsonl`、`queries.jsonl` 及 `qrels/` 子目录。 | **必需**（二选一逻辑，通过 `resolve_dataset_dir` 解析） |
| `--embedding-dir` 或 `--embedding-root` | 目录路径参数 | 文件系统路径 | N/A | 预计算嵌入文件的根目录或直接路径。通常与构建索引时使用的嵌入目录一致。 | **必需**（二选一逻辑，通过 `resolve_embedding_dir` 解析） |
| **（隐式）** `query_points.npy` <br>（位于上述嵌入目录） | `.npy` 文件 | NumPy 二进制格式（`mmap` 读取） | `(num_query_tokens, dim)`，`float32` | 所有查询的 token 级嵌入（已展平，按查询顺序拼接）。 | **必需** |
| **（隐式）** `query_offsets.npy` <br>（位于上述嵌入目录） | `.npy` 文件 | NumPy 二进制格式 | `(num_queries + 1,)`，`int64` | 每个查询在 `query_points.npy` 中的起始和结束索引，用于切分单个查询的嵌入。 | **必需** |
| **（隐式）** 已构建的 WARP 索引目录<br>（由 `--index-root` + `--index` 或自动名称定位） | 目录 | 文件系统目录 | N/A | 包含 `metadata.json`、`codec` 文件、分块文件及 IVF 倒排文件的完整索引目录。由 `index_from_embeddings.py` 生成。 | **必需** |
| `--index-root` | 目录路径参数 | 文件系统路径 | N/A | 索引存储的根目录。脚本会在该目录下查找 `index_name` 子目录。 | **必需** |
| `--index` | 字符串参数 | 命令行字符串 | N/A | 自定义索引名称。若不指定，则根据 `dataset`、`encoder`、`nbits` 自动生成。 | 可选 |
| `--encoder` | 字符串参数 | 命令行字符串 | N/A | 编码器名称，用于生成默认索引名及基线名称。 | 可选（默认 `colbert`） |
| `--nbits` | 整数参数（1/2/4/8） | 命令行整数 | N/A | 索引压缩时使用的比特数，需与待测索引一致。 | 可选（默认 `2`） |
| `--split` | 字符串参数 | 命令行字符串（如 `dev`/`test`） | N/A | 数据切分名称。若不指定，则根据 `dataset` 自动推断（`msmarco` → `dev`，其余 → `test`）。 | 可选 |
| `--ncells` | 整数列表参数 | 命令行整数列表（空格分隔） | N/A（如 `1 2 4 8`） | 待扫描的 `ncells` 值列表（即检索时访问的聚类中心数量）。 | 可选（默认 `[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]`） |
| `--k` | 整数参数 | 命令行整数 | N/A | 检索时返回的候选文档数（`scorer.rank` 的 Top-K 截断）。 | 可选（默认 `100`） |
| `--metrics-k` | 整数列表参数 | 命令行整数列表（空格分隔） | N/A（如 `10 100`） | 计算评估指标（NDCG、Recall 等）时使用的多个截断值。 | 可选（默认 `[10, 100]`） |
| `--max-queries` | 整数参数 | 命令行整数 | N/A | 限制参与评估的查询数量（仅使用 BEIR/qrels 顺序中的前 N 个查询），用于快速测试。 | 可选 |
| `--centroid-score-threshold` | 浮点数参数 | 命令行浮点数 | N/A | 检索时的聚类中心分数阈值（透传给 `ColBERTConfig`，用于过滤低质量中心）。 | 可选 |
| `--ndocs` | 整数参数 | 命令行整数 | N/A | 检索时每个中心最多取出的文档数（透传给 `ColBERTConfig`）。 | 可选 |
| `--baseline` | 字符串参数 | 命令行字符串 | N/A | 输出 CSV 中的基线名称（用于实验标记）。若不指定，自动生成为 `warp_precomputed_{encoder}_nbits{nbits}`。 | 可选 |
| `--output-dir` | 目录路径参数 | 文件系统路径 | N/A | CSV 报告的输出目录。脚本会按 `{dataset}.split={split}.baseline={baseline}.ncells_sweep.csv` 格式自动命名。 | 可选（默认 `REPO_ROOT/resutls`，注意原文拼写为 `resutls`） |
| `--output-csv` | 文件路径参数 | 文件系统路径 | N/A | 显式指定 CSV 输出文件的完整路径（若提供，则忽略 `--output-dir` 的自动命名逻辑）。 | 可选 |
| `--append` | 布尔标志参数 | 命令行开关（无需值） | N/A | 若 CSV 文件已存在，追加新行而非覆盖。 | 可选（默认关闭） |
| `--tsv-output` | 文件路径参数 | 文件系统路径 | N/A | 可选输出路径，用于保存 TREC 格式的检索结果 TSV 文件。**注意：使用该参数时必须确保 `--ncells` 只指定一个值**，否则报错。 | 可选 |
| **（隐式）** BEIR 数据集文件<br>（位于 `dataset_dir` 下） | `.jsonl` / `.tsv` 文件 | UTF-8 文本 | 标准 BEIR 格式 | 包含 `corpus.jsonl`（文档）、`queries.jsonl`（查询）、`qrels/{split}.tsv`（标注）。用于提取真实文档 ID 和计算指标。 | **必需** |
| **（隐式）** 环境变量 `TORCH_NUM_THREADS` | 环境变量 | 整数字符串 | N/A | 若设置，则通过 `torch.set_num_threads` 配置 PyTorch 线程数。 | 可选 |

---

**输出（程序运行结果与生成文件）**

| 输出目标 | 类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CSV 报告文件**（`--output-csv` 或自动生成路径） | 数据汇总文件 | UTF-8 文本，逗号分隔（CSV） | **每行对应一个 `ncells` 配置**，包含以下列：<br>- 元数据：`dataset`、`split`、`baseline`、`index_name`、`ncells`、`num_queries`、`elapsed_sec`、`qps` 等。<br>- 评估指标：`ndcg_10`、`ndcg_100`、`map_10`、`map_100`、`recall_10`、`recall_100`、`p_10`、`p_100`、`mrr_10`、`mrr_100` 等（根据 `--metrics-k` 动态生成）。 | 批量实验汇总，便于横向对比不同 `ncells` 对检索质量（NDCG/Recall）和速度（QPS）的影响。 |
| **TSV 检索结果文件**（仅当指定 `--tsv-output` 且 `ncells` 唯一时生成） | 排序结果文件 | UTF-8 文本，制表符分隔（TSV） | `(num_queries * k, 4)` 列：<br>`query-id`, `corpus-id`, `rank`, `score` | 标准 TREC 格式的检索结果，包含每个查询最终返回的文档排名及得分。 | 可用于后续更细粒度的分析、重排序或与其他检索器输出进行公平对比。 |
| **标准输出（控制台）** | 进度日志与摘要 | 纯文本 | 多行字符串 | 打印每个 `ncells` 值的运行结果摘要（如 `ncells=8 NDCG@10=0.342 ... elapsed=5.2s`），以及最终保存路径。 | 实时监控扫描进度，快速查看关键指标变化趋势。 |
| **CSV 文件（追加模式）** | 已存在文件的修改 | N/A | N/A | 若使用 `--append` 且文件存在，则在文件末尾新增行，保留历史记录。 | 支持多次运行合并结果，便于累积实验数据。 |

---

## IV. Evaluation Metrics

| 指标分类 | CSV 列名模式 | 是否受 `--metrics-k` 控制 | 计算公式 / 计算方法 | 核心说明与值域 |
| :--- | :--- | :--- | :--- | :--- |
| **检索质量** <br>（NDCG） | `ndcg_{k}` <br>（如 `ndcg_10`） | ✅ 是 | ① 计算 **DCG@K** = \(\sum_{i=1}^{K} \frac{2^{rel(d_i)} - 1}{\log_2(i+1)}\) <br> ② 计算 **IDCG@K**（理想排序下的 DCG 最大值）<br> ③ **NDCG@K** = DCG@K / IDCG@K | **归一化折损累计增益**。<br>衡量排序质量，对排名靠前的高相关文档给予更高权重。<br>值域 [0, 1]，越高越好。<br>若某查询无相关文档（IDCG=0），该查询 NDCG 计为 0。 |
| **检索质量** <br>（MAP） | `map_{k}` <br>（如 `map_100`） | ✅ 是 | ① 对单查询计算 **AP@K** = \(\frac{1}{\min(\|R_q\|, K)} \sum_{i=1}^{K} (P@i \times rel(d_i))\) <br> ② **MAP@K** = \(\frac{1}{\|Q\|} \sum_{q \in Q} AP@K(q)\) | **平均精度均值**。<br>综合考察 Precision 和 Recall，对相关文档在结果列表中的位置敏感。<br>值域 [0, 1]，越高越好。<br>无相关文档的查询 AP=0。 |
| **检索质量** <br>（Recall） | `recall_{k}` <br>（如 `recall_100`） | ✅ 是 | **Recall@K** = \(\frac{\|T_q \cap R_q\|}{\|R_q\|}\) | **召回率**。<br>衡量前 K 个结果能覆盖多少全部相关文档。<br>值域 [0, 1]，越高越好。<br>无相关文档时 Recall=0。 |
| **检索质量** <br>（Precision） | `p_{k}` <br>（如 `p_10`） | ✅ 是 | **P@K** = \(\frac{\|T_q \cap R_q\|}{K}\) | **精确率**。<br>衡量前 K 个结果中相关文档的密度。<br>值域 [0, 1]，越高越好。<br>注意分母固定为 K（与总相关文档数无关）。 |
| **检索质量** <br>（MRR） | `mrr_{k}` <br>（如 `mrr_10`） | ✅ 是 | ① 找首个相关文档在结果列表中的位置 \(rank_{first}\)<br> ② **RR@K** = \(1 / rank_{first}\)（若 \(rank_{first} \le K\) 且存在相关文档，否则为 0）<br> ③ **MRR@K** = \(\frac{1}{\|Q\|} \sum_{q \in Q} RR@K(q)\) | **平均倒数排名**。<br>只关心第一个命中的位置，常用于问答等任务。<br>值域 [0, 1]，越高越好。 |
| **检索效率** <br>（总耗时） | `elapsed_sec` | ❌ 否 | 使用 Python `time.perf_counter()` 在检索循环前后做差，得到 **总耗时（秒）**。 | 仅包含所有查询调用 `scorer.rank` 的纯检索耗时。<br>不含加载索引、计算指标、I/O 写入的时间。<br>数值越小越好。 |
| **检索效率** <br>（吞吐量） | `qps` | ❌ 否 | **QPS** = 查询总数（`num_queries`）/ `elapsed_sec` | **每秒查询数**。<br>衡量检索系统的吞吐能力。<br>数值越大越好。 |

---

**补充约定（便于理解表格）**：
- \( R_q \)：查询 \( q \) 在 Ground Truth（qrels）中的全部相关文档集合。
- \( T_q \)：检索系统返回的前 \( K \) 个文档集合（按得分降序）。
- \( rel(d_i) \)：文档 \( d_i \) 的相关性等级（BEIR 中通常为 0/1 或多级整数）。
- \( P@i \)：前 \( i \) 个检索结果的精确率（Precision）。
- `--metrics-k` 参数控制实际生成哪些 K 值的列（如只指定 `[10, 100]`，则只输出 `@10` 和 `@100` 版本）。