# flat multivector

## I. environment

```bash
conda create -n flatmulti python=3.10.18 pip setuptools wheel -y
conda activate flatmulti

pip install -r requirements.txt
```

```bash
mkdir -p /data1/chenyifeng/libs
cd /data1/chenyifeng/libs

git clone https://github.com/cmuparlay/parlaylib.git
cd parlaylib

mkdir build && cd build

cmake -DCMAKE_INSTALL_PREFIX=/data1/chenyifeng/libs/parlay_install ..
make -j
make install
```

```bash
rm -rf /data1/chenyifeng/multi-vector-retrieval/build/*
rm /data1/chenyifeng/MultiVector-Backup/multivector_baselines/script/evaluation/IGP*.so

export CMAKE_PREFIX_PATH=/data1/chenyifeng/libs/parlay_install:$CMAKE_PREFIX_PATH
export Parlay_DIR=/data1/chenyifeng/libs/parlay_install/share/parlay/cmake
```



## II. run

### 1. Prepare a flat dataset: `script/flat_multivector/prepare_flat_multivector_dataset.py`

    ```bash
    python script/flat_multivector/prepare_flat_multivector_dataset.py \
        --username <username> \
        --dataset <new-dataset-name> \
        --doc-embeddings <doc-embeddings.npy> \
        --doc-lens <doc-lens.npy> \
        --query-embeddings <query-embeddings.npy> \
        --query-lens <query-lens.npy> \
        --groundtruth-ivecs <groundtruth.ivecs> \
        --force
    ```

    Example for ```scidocs-large-multi``` :

    ```bash
    python script/flat_multivector/prepare_flat_multivector_dataset.py \
        --username chenyifeng \
        --dataset scidocs-large-multi-flat-test \
        --doc-embeddings ./data/scidocs-large-multi/full_multi_embeddings_scidocs-large.npy \
        --doc-lens ./data/scidocs-large-multi/full_multi_chunk_num_scidocs-large.npy \
        --query-embeddings ./data/scidocs-large-multi/full_multi_embeddings_scidocs-large_query.npy \
        --query-lens ./data/scidocs-large-multi/full_multi_chunk_num_scidocs-large_query.npy \
        --max-queries 1000 \
        --force
    ```

### 2. Build and run Plaid

There are two ways to do this.

- Option A: explicit commands

    Build the Plaid index:

    ```bash
    python script/flat_multivector/build_plaid_from_flat_dataset.py \
        --username <username> \
        --dataset <dataset name> \
        --manifest <manifest.json path>
    ```

    Run Plaid retrieval:

    ```bash
    python script/evaluation/eval_plaid.py \
        --username <username> \
        --dataset <dataset name>
    ```

    Evaluate Plaid results against local ground truth:

    ```bash
    python script/flat_multivector/eval_flat_groundtruth.py \
        --username <username> \
        --dataset <dataset name> \
        --method plaid
    ```

    Example for ```scidocs-large-multi``` :

    ```
    python script/flat_multivector/build_plaid_from_flat_dataset.py \
        --username chenyifeng \
        --dataset scidocs-large-multi-flat-test \
        --manifest /data1/chenyifeng/Dataset/multi-vector-retrieval/FlatData/scidocs-large-multi-flat-test/manifest.json
    ```

    ```bash
    python script/evaluation/eval_plaid.py \
        --username chenyifeng \
        --dataset scidocs-large-multi-flat-test
    ```

    ```bash
    python script/flat_multivector/eval_flat_groundtruth.py \
        --username chenyifeng \
        --dataset scidocs-large-multi-flat-test \
        --method plaid
    ```

- Option B: one wrapper

    ```bash
    bash script/flat_multivector/run_plaid_flat.sh <username> <dataset name>
    ```

    Example for ```scidocs-large-multi``` :

    ```
    bash script/flat_multivector/run_plaid_flat.sh chenyifeng scidocs-large-multi-flat-test
    ```

### 3. Run Dessert

Important:

- Dessert depends on Plaid artifacts
- run Plaid build first
- Dessert uses the Plaid centroids and codes

Command:

```bash
python script/evaluation/eval_dessert.py \
  --username <username> \
  --dataset <dataset name>
```

Evaluate Dessert:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username <username> \
  --dataset <dataset name> \
  --method dessert
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_dessert_flat.sh <username> <dataset name>
```


Example for ```scidocs-large-multi``` :

```bash
python script/evaluation/eval_dessert.py \
  --username chenyifeng \
  --dataset scidocs-large-multi-flat-test
```

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username chenyifeng \
  --dataset scidocs-large-multi-flat-test \
  --method dessert
```

Or

```bash
bash script/flat_multivector/run_dessert_flat.sh chenyifeng scidocs-large-multi-flat-test
```


### 4. Run MUVERA

Build environment:

```bash
mkdir -p local

# fmt
git clone https://github.com/fmtlib/fmt.git
cd fmt
git checkout 9.1.0
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/data1/chenyifeng/MultiVector-Backup/multivector_baselines/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DFMT_TEST=OFF \
    -DBUILD_SHARED_LIBS=ON   
make -j $(nproc)
make install
cd ../..

# spdlog
git clone https://github.com/gabime/spdlog.git
cd spdlog
git checkout v1.11.0
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/data1/chenyifeng/MultiVector-Backup/multivector_baselines/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPDLOG_FMT_EXTERNAL=ON \
    -DSPDLOG_BUILD_SHARED=ON \
    -Dfmt_DIR=/data1/chenyifeng/MultiVector-Backup/multivector_baselines/local/lib/cmake/fmt
make -j $(nproc)
make install

export LD_LIBRARY_PATH=/data1/chenyifeng/MultiVector-Backup/multivector_baselines/local/lib:$LD_LIBRARY_PATH
```

Command:

```bash
python script/evaluation/eval_muvera.py \
  --username <username> \
  --dataset_name <dataset name>
```

Important:

- MUVERA reads `Embedding/<dataset>/base_embedding/*.npy`
- those files are created by the Plaid build path
- so if you only ran the preparation script, run Plaid build first

Evaluate MUVERA:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username <username> \
  --dataset <dataset name> \
  --method MUVERA
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_muvera_flat.sh <username> <dataset name>
```

Example for ```scidocs-large-multi``` :

```bash
python script/evaluation/eval_muvera.py \
  --username chenyifeng \
  --dataset_name scidocs-large-multi-flat-test
```

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username chenyifeng \
  --dataset scidocs-large-multi-flat-test \
  --method MUVERA
```

Or

```bash
bash script/flat_multivector/run_muvera_flat.sh chenyifeng scidocs-large-multi-flat-test
```



### 5. Run IGP

Command:

```bash
python script/evaluation/eval_igp.py \
  --username <username> \
  --dataset_name <dataset name>
```

Important:

- IGP also reads `Embedding/<dataset>/base_embedding/*.npy`
- those files are created by the Plaid build path
- so if you only ran the preparation script, run Plaid build first

Evaluate IGP:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username <username> \
  --dataset <dataset name> \
  --method IGP
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_igp_flat.sh <username> <dataset name>
```


Example for ```scidocs-large-multi``` :

```bash
python script/evaluation/eval_igp.py \
  --username chenyifeng \
  --dataset_name scidocs-large-multi-flat-test
```

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username chenyifeng \
  --dataset scidocs-large-multi-flat-test \
  --method IGP
```

Or

```bash
bash script/flat_multivector/run_igp_flat.sh chenyifeng scidocs-large-multi-flat-test
```
''


## III. Notes

1. `MultiVector-Backup/multivector_baselines/script/flat_multivector/prepare_flat_multivector_dataset.py` ：


**脚本输入文件一览表 (Inputs)**

| 命令行参数 | 文件类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--doc-embeddings` | `.npy` (NumPy数组) | 二进制 | `(总文档向量数, 维度)` <br> dtype: `float32` | 整个语料库（Corpus）中所有文档的Token/段落级嵌入矩阵。 | **必需** |
| `--doc-lens` | `.npy` (NumPy数组) | 二进制 | `(文档总数,)` <br> dtype: `int64` | 每个文档所包含的向量个数（即Token或段落数量）。所有元素之和必须等于 `doc_embeddings` 的行数。 | **必需** |
| `--query-embeddings` | `.npy` (NumPy数组) | 二进制 | `(总查询向量数, 维度)` <br> dtype: `float32` | 所有查询的Token/段落级嵌入矩阵。 | **必需** |
| `--query-lens` | `.npy` (NumPy数组) | 二进制 | `(查询总数,)` <br> dtype: `int64` | 每个查询所包含的向量个数。所有元素之和必须等于 `query_embeddings` 的行数。 | **必需** |
| `--groundtruth-ivecs` | `.ivecs` (二进制整数向量) | 二进制 (int32) | `(查询总数, 可变K)` <br> 每行首字段为维度K | 标准ANN基准格式的真值文件。每个查询对应一行，存储最近邻的文档ID列表（ID可为-1填充）。 | 可选（三选一） |
| `--local-qrels-tsv` | `.tsv` (制表符分隔) | 文本 (UTF-8) | 每行3列：<br>`qid(整数) \t pid(整数) \t score(浮点)` | 本地整数索引的qrels文件。要求qid和pid必须已在0~N-1范围内。脚本会过滤 `score <= 0` 的行。 | 可选（三选一） |
| `--qrels-tsv` | `.tsv` (制表符分隔) | 文本 (UTF-8) | 每行至少2列：<br>`query_id(字符串) \t corpus_id(字符串) \t [score]` | 外部字符串ID的qrels文件。需配合 `--query-ids-json` 和 `--corpus-ids-json` 进行映射。 | 可选（三选一） |
| `--query-ids-json` | `.json` (JSON列表) | 文本 (UTF-8) | `["id_1", "id_2", ...]` | 外部查询字符串ID列表，其列表顺序（索引位置）对应本地整数ID（0, 1, 2...）。 | 使用 `--qrels-tsv` 时**必需** |
| `--corpus-ids-json` | `.json` (JSON列表) | 文本 (UTF-8) | `["id_1", "id_2", ...]` | 外部文档字符串ID列表，其列表顺序（索引位置）对应本地整数ID（0, 1, 2...）。 | 使用 `--qrels-tsv` 时**必需** |
| `--runtime-root` | (目录路径) | 系统路径 | - | 运行时根目录。若未指定，默认为 `/data1/{username}/Dataset/multi-vector-retrieval`。 | 可选 |
| `--flat-root` | (目录路径) | 系统路径 | - | 存放处理后的FlatData的根目录。若未指定，默认为 `{runtime_root}/FlatData`。 | 可选 |

> **注**：`--groundtruth-ivecs`、`--local-qrels-tsv`、`--qrels-tsv` 三者必须且只能提供一个，否则将生成空的标准答案（所有查询的 `answer_pids` 为空）。

---

**脚本输出文件一览表 (Outputs)**

| 输出路径（相对于数据集根目录） | 文件类型 | 格式 / 编码 | 内容描述 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| **`FlatData/{dataset}/manifest.json`** | `.json` (JSON对象) | 文本 (UTF-8) | 包含所有输入/输出路径、数据集统计信息（文档数、向量数、维度等）和分片数量的元数据清单。 | 供后续流程或人工检查数据版本与路径。 |
| **`FlatData/{dataset}/doc_embeddings/transformed_embeddings/doc_count`** | PyTorch 存储文件 | 二进制 (`torch.save`) | 存储文档总数（一个整数）。 | 加载分片时确认文档总数。 |
| **`FlatData/{dataset}/doc_embeddings/transformed_embeddings/embeddings.{i}.pt`** | PyTorch 张量文件 | 二进制 (`torch.save`) | 每个分片保存一个元组：<br>1. `batch_tensor`：形状 `(该分片总向量数, 维度)`<br>2. `batch_doc_lens`：该分片内每个文档的向量个数列表。 | 将超大文档嵌入拆分为多个小文件，便于内存映射和分布式加载。 |
| **`FlatData/{dataset}/query_embeddings/transformed_embeddings/query_n_vec_length.npy`** | `.npy` (NumPy数组) | 二进制 | `(有效查询数,)` <br> dtype: `int64` | 处理后的查询长度向量（如果指定了 `--max-queries`，则已被截断）。 | 检索时获取每个查询的向量个数。 |
| **`FlatData/{dataset}/query_groundtruth/queries.gnd.jsonl`** | `.jsonl` (JSON Lines) | 文本 (UTF-8) | 每行一个JSON对象：<br>`{"qid": 整数, "answer_pids": [整数列表]}`<br>列表已去重。 | 统一格式的标准答案（Ground Truth），作为后续生成评估文件的源数据。 |
| **`RawData/{dataset}/document/collection.tsv`** | `.tsv` (制表符分隔) | 文本 (UTF-8) | 共 `n_doc` 行，每行仅包含一个整数：`doc_id`（0 到 n_doc-1）。 | 模拟Pyserini等标准检索工具的Collection格式，用于构建索引。 |
| **`RawData/{dataset}/document/queries.dev.tsv`** | `.tsv` (制表符分隔) | 文本 (UTF-8) | 共 `n_query` 行，每行仅包含一个整数：`query_id`（0 到 n_query-1）。 | 模拟标准查询集格式，用于检索时读取待查询列表。 |
| **`RawData/{dataset}/document/transformed_embeddings`** | **符号链接 (Symlink)** | 指向 `FlatData/.../transformed_embeddings` | 指向文档分片目录的软链接。 | 保持RawData目录下结构统一，实际数据存储在FlatData中。 |
| **`RawData/{dataset}/document/queries.gnd.jsonl`** | **符号链接 (Symlink)** | 指向 `FlatData/.../queries.gnd.jsonl` | 指向标准答案JSONL的软链接。 | 同上，方便统一访问。 |
| **`Embedding/{dataset}/{dataset}-groundtruth-top10--.tsv`** | `.tsv` (制表符分隔) | 文本 (UTF-8) | 4列：<br>`qid \t pid \t rank \t 1`<br>仅包含每个查询的前10个相关文档。 | Plaid/DESSERT官方评估脚本所需的特定格式真值（Top-10）。 |
| **`Embedding/{dataset}/{dataset}-groundtruth-top100--.tsv`** | `.tsv` (制表符分隔) | 文本 (UTF-8) | 4列：<br>`qid \t pid \t rank \t 1`<br>仅包含每个查询的前100个相关文档。 | Plaid/DESSERT官方评估脚本所需的特定格式真值（Top-100）。 |

> **注**：所有输出目录在生成前都会自动创建（`ensure_dir`）。如果指定了 `--force`，脚本会先删除已存在的输出目录再重新生成。

---


2. `MultiVector-Backup/multivector_baselines/script/flat_multivector/build_plaid_from_flat_dataset.py`:

**输入（Inputs）**

运行 `build_plaid_from_flat_dataset.py` 所需的所有输入数据。其中命令行参数直接传递，而数据文件路径由 `manifest.json` 提供。

| 输入来源 | 类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **命令行 `--username`** | 字符串 | 文本 | N/A | 用户标识，用于构造数据根目录（`/data1/{username}/...`） | **必需** |
| **命令行 `--dataset`** | 字符串 | 文本 | N/A | 数据集名称，用于定位子目录及命名输出文件 | **必需** |
| **命令行 `--manifest`** | 文件路径 | JSON (UTF-8) | 嵌套字典 (`dict`) | 包含数据集所有预处理产物的元数据映射（路径、统计信息等） | 可选（脚本会自动拼接默认路径） |
| **manifest["prepared"]["doc_transformed_embeddings"]** | 文件夹 | 二进制 (`.npy` 或 `.pt`) | 多个文件，总体形状为 `[总文档Token数, 维度]` | 预计算好的文档多向量扁平化嵌入（2D 扁平存储） | **必需** |
| **manifest["prepared"]["groundtruth_jsonl"]** | 文件 | JSONL (UTF-8) | 每行一个 JSON 对象：`{"qid":..., "docid":..., "relevance":...}` | 查询-文档相关性标注数据，用于后续评估 | **必需** |
| **manifest["prepared"]["doc_count_file"]** | 文件 | PyTorch 张量 (`.pt`) 或 NumPy (`.npy`) | 形状 `(n_docs,)`，元素为 `int` | 记录每个文档包含多少个 Token 向量（用于索引分块） | **必需** |
| **manifest["prepared"]["prepared_query_lens"]** | 文件 | NumPy (`.npy`) | 形状 `(n_queries,)`，元素为 `int` | 记录每个查询的实际 Token 长度（用于重组扁平查询嵌入） | **必需** |
| **manifest["source"]["query_embeddings"]** | 文件 | NumPy (`.npy`) 或 PyTorch (`.pt`) | 2D `[总查询Token数, 维度]` 或 3D `[n_queries, max_len, 维度]` | 预计算的查询嵌入（扁平或填充格式） | **必需** |
| **manifest["counts"]["dim"]** | 整数字段 | JSON 数字 | N/A | 嵌入向量的维度（如 128） | **必需** |
| **manifest["counts"]["n_query"]** | 整数字段 | JSON 数字 | N/A | 查询的总数量（用于校验长度文件） | **必需** |

---

**输出（Outputs）**

运行脚本后，在 `/data1/{username}/Dataset/multi-vector-retrieval/` 下生成的所有产物。

| 关联命令行参数 | 文件类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--dataset`<br>`--username` | **Plaid 索引目录** (`Index/{dataset}/plaid/`) | 二进制 + 自定义 ColBERT 格式 | 包含 `ivf.index`、`codes.pt`、`metadata.json` 等 | 完整的 Plaid 可检索索引（含 IVF 倒排、量化码本、文档ID映射）。是检索系统的核心。 | **必需**（检索依赖） |
| `--dataset`<br>`--username` | **文档长度文件** (`Embedding/{dataset}/doclens.npy`) | NumPy (`.npy`) | 形状 `(n_docs,)`，`int32` 或 `int64` | 合并所有分块后，每个文档对应的向量个数。用于检索时的分数归一化。 | **必需**（检索依赖） |
| `--dataset`<br>`--username` | **查询长度文件** (`Embedding/{dataset}/query_n_vec_length.npy`) | NumPy (`.npy`) | 形状 `(n_queries,)`，`int32` 或 `int64` | 每个查询的 Token 长度（从 `prepared_query_lens` 复制）。用于检索时处理变长查询。 | **必需**（检索依赖） |
| `--dataset`<br>`--username` | **查询嵌入文件** (`Embedding/{dataset}/query_embedding.npy`) | NumPy (`.npy`) | 3D 形状 `[n_queries, max_len, dim]`，`float32` | 加载 `query_embeddings` 后重组为填充齐整的 3D 张量。避免重复编码，加速检索。 | **必需**（检索依赖） |
| `--dataset`<br>`--username` | **Plaid Groundtruth TSV** (`Embedding/{dataset}/queries.tsv` 及 `qrels.tsv`) | TSV (UTF-8) | 表格数据，`qid` / `docid` / `relevance` 列 | 由 `write_plaid_groundtruth_tsvs` 从 JSONL 转换而来，用于 Plaid 官方评估工具计算 Recall/MRR。 | **必需**（评估依赖） |
| `--dataset`<br>`--username` | **构建性能日志** (`Result/performance/{dataset}-build_index-plaid-.json`) | JSON (UTF-8) | 字典：`{"build_index_time (s)": float, "encode_passage_time (s)": float}` | 记录索引构建总耗时和文档编码耗时（若使用预计算嵌入，后者为 0）。 | 可选（仅性能分析） |
| `--dataset`<br>`--username` | **查询编码性能日志** (`Result/performance/{dataset}-encode_query.json`) | JSON (UTF-8) | 字典：`{"total_encode_time_ms": float, "n_encode_query": int, ...}` | 记录预计算查询嵌入的加载/编码耗时。 | 可选（仅性能分析） |
| `--dataset`<br>`--username` | **符号链接（辅助）**<br>(`RawData/{dataset}/document/transformed_embeddings` 及 `queries.gnd.jsonl`) | 软链接 (Symlink) | N/A | 指向原始嵌入文件夹和 Groundtruth 的软链接，便于 ColBERT 内部统一路径解析。 | 可选（仅为兼容性） |

---

3. `MultiVector-Backup/multivector_baselines/script/evaluation/eval_plaid.py`

**输入（Inputs）**

运行 `eval_plaid.py` 所需的所有依赖项。其中“命令行参数”直接传递用户标识，而“实验配置”在脚本内部定义（或通过外部传参），其余为必须已存在的数据文件。

| 输入来源 / 关联参数 | 文件类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **命令行 `--username`**<br>**命令行 `--dataset`** | 字符串参数 | 文本 | N/A | 用户标识和数据集名称，用于构造所有数据路径（如 `/data1/{username}/.../{dataset}/`） | **必需** |
| **`Index/{dataset}/plaid/`**<br>（由构建脚本生成） | 索引目录 | 二进制 + ColBERT 自定义格式 | 包含 `ivf.index`、`codes.pt`、`metadata.json` 等文件 | 完整的 Plaid 可检索索引（含 IVF 倒排列表、乘积量化码本、文档偏移映射） | **必需**（检索核心） |
| **`Embedding/{dataset}/query_embedding.npy`**<br>（由构建脚本生成） | NumPy 数组文件 | 二进制 (`.npy`)，小端序 | 3D 浮点数组：`[n_queries, max_len, dim]`，`float32` | 预计算并填充好的查询嵌入张量（变长查询已填充 0 至等长）。检索时直接加载，避免重复编码。 | **必需**（查询向量来源） |
| **`RawData/{dataset}/document/queries.dev.tsv`**<br>（由构建脚本生成或原始提供） | TSV 文本文件 | UTF-8 编码，`\t` 分隔 | 通常为 `[qid, query_text]` 两列，但脚本仅读取第一列作为 `qid` | 查询 ID 列表文件。脚本遍历该文件获取 `qid`，用于对齐最终输出的排名结果。 | **必需**（输出排名时需对应 qid） |
| **实验配置参数**<br>（来自 `config_l` 或网格搜索） | Python 字典/列表 | 内存对象 | 整型 / 浮点型 | 包含 `topk`（返回文档数）、`ndocs`（IVF 候选数）、`ncells`（聚类中心数）、`centroid_score_threshold`（阈值）、`n_thread`（CPU 线程数）。用于控制检索行为。 | **必需**（决定检索策略） |

---

**输出（Outputs）**

运行脚本后，针对**每一组检索参数组合**（网格搜索会产生多组），在 `/data1/{username}/Dataset/multi-vector-retrieval/` 下生成以下文件。

| 关联命令行参数 | 文件类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--dataset`<br>`--username`<br>（文件名含 `topk` 及检索参数后缀） | **排名结果文件**<br>（`Result/answer/{dataset}-plaid-top{topk}-{suffix}.tsv`） | TSV 文本文件 | 表格结构，通常包含 4 列：<br>`[qid, docid, rank, score]`，每行一个检索结果 | 存储每个查询检索出的前 `topk` 个文档的 ID、排名序号和相似度分数。用于后续计算 Recall/MRR 或人工检查。 | **必需**（检索最终产物） |
| `--dataset`<br>`--username`<br>（文件名含 `topk` 及检索参数后缀） | **性能日志文件**<br>（`Result/performance/{dataset}-retrieval-plaid-top{topk}-{suffix}.json`） | JSON 文本文件 | 嵌套字典，包含：<br>- `n_query`（查询总数）<br>- `topk`<br>- `retrieval`（参数配置）<br>- `search_time`（平均/百分位数耗时、IVF/Filter/Refine 阶段平均耗时、平均精排向量数等） | 详细记录本次检索的时间性能指标。可用于分析不同参数对速度的影响，是实验调优的核心数据。 | **必需**（性能分析依赖） |

---

4. `MultiVector-Backup/multivector_baselines/script/flat_multivector/eval_flat_groundtruth.py`

**输入（Inputs）**

| 输入来源 / 参数 | 类型 | 格式/编码 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- |
| `--username`、`--dataset` | 字符串参数 | 文本 | 用于构造根路径（`/data1/{username}/Dataset/multi-vector-retrieval/`） | **必需** |
| `--manifest`（或自动拼接） | JSON 文件 | UTF-8 | 包含 `prepared.groundtruth_jsonl` 字段，指向 Groundtruth 文件路径 | **必需** |
| `--method`（可选） | 字符串参数 | 文本 | 限定方法名，筛选答案文件 | 可选 |
| `--k-values` | 整数列表 | 命令行参数 | 评估的 top-k 值，如 `10 100` | 默认 `[10,100]` |
| `groundtruth_jsonl`（从 manifest 获取） | JSONL 文件 | UTF-8，每行一个 JSON 对象 | 包含 `qid` 和 `answer_pids` 数组的标注数据 | **必需** |
| `Result/answer/{dataset}-*.tsv` | TSV 文件 | UTF-8，`\t` 分隔 | 检索排名结果，通常 4 列：`qid, pid, rank, score` | **必需**（至少一个） |
| `Result/performance/{dataset}-retrieval-*.json` | JSON 文件 | UTF-8 | 对应检索实验的性能日志，包含 `n_query`、`topk`、`search_time` 等字段 | **必需**（与每个答案文件配对） |

**输出（Outputs）**

| 输出文件 | 类型 | 格式/编码 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- |
| `Result/performance/{dataset}-flat-eval-summary.csv`（或自定义路径） | CSV 文件 | UTF-8，逗号分隔 | 综合汇总表，每行对应一次检索实验（一个答案文件），列包括：<br> - 文件标识（`answer_file`, `performance_file`）<br> - `method`, `n_query`, `topk`, `qps`, 时间指标<br> - 构建和检索配置参数（动态列）<br> - 对每个 k：`recall@{k}`, `mrr@{k}`, `success@{k}`, `ndcg@{k}` | **必需**（评估最终产物） |

---

5. `MultiVector-Backup/multivector_baselines/script/evaluation/eval_dessert.py`


**输入（Inputs）**

| 关联参数 / 来源 | 文件 / 目录类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **命令行 `--username`**<br>**命令行 `--dataset`** | 字符串参数 | 文本 | N/A | 用户标识和数据集名称，用于构造所有路径（如 `/data1/{username}/Dataset/multi-vector-retrieval/.../{dataset}/`） | **必需** |
| **命令行 `--host_name`** | 字符串参数 | 文本 | N/A | 决定使用 `dbg` 或 `local` 配置，控制网格搜索参数范围（但具体数值在脚本内固定，不影响文件输入） | 可选（默认 `local`） |
| **Plaid 索引目录**<br>`Index/{dataset}/plaid/` | 目录 | 二进制（PyTorch 张量 `.pt`） | – | 包含 `centroids.pt`（聚类中心矩阵）和 `{chunkID}.codes.pt`（每个向量的质心分配 ID）。用于提取 Dessert 所需的基础数据。 | **必需**（由 `build_plaid_from_flat_dataset.py` 生成） |
| **文档嵌入目录**<br>`Embedding/{dataset}/base_embedding/` | 目录 | NumPy 二进制 (`.npy`) | 每个分块 `encoding{i}_float32.npy` 形状为 `[n_vectors_in_chunk, dim]`，`float32` | 文档 Token 向量的分块存储。用于在构建 Dessert 索引时按文档读取嵌入向量。 | **必需**（由构建脚本生成） |
| **文档长度文件**<br>`Embedding/{dataset}/doclens.npy` | NumPy 文件 | 二进制 (`.npy`) | 形状 `(n_docs,)`，`int` 类型 | 每个文档包含的 Token 向量数量。用于计算文档的起始偏移量，以正确拆分嵌入文件中的向量。 | **必需**（由构建脚本生成） |
| **查询嵌入文件**<br>`Embedding/{dataset}/query_embedding.npy` | NumPy 文件 | 二进制 (`.npy`) | 3D 浮点数组 `[n_queries, max_len, dim]`，`float32` | 预计算并填充好的查询嵌入张量。检索时直接加载。 | **必需**（由构建脚本生成） |
| **查询 ID 文件**<br>`RawData/{dataset}/document/queries.dev.tsv` | TSV 文件 | UTF-8，`\t` 分隔 | 至少两列：`qid` 和 `query_text` | 用于获取查询 ID 列表，以便在输出排名文件中使用原始 `qid`（而非内部索引）。若文件不存在，则使用 `0..n_query-1` 作为替代。 | **可选**（若缺失则使用索引顺序） |

---

**输出（Outputs）**

运行 `eval_dessert.py` 后，根据网格搜索配置（不同的 `n_table` 和检索参数组合），生成以下文件：

| 关联参数 | 文件 / 目录类型 | 格式 / 编码 | 数据结构 / 形状 | 内容描述 | 必需性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--username`<br>`--dataset`<br>构建参数 `n_table` | **Dessert 索引文件**<br>`Index/{dataset}/dessert/dessert-{dataset}-n_table_{num_tables}.index` | 二进制（由 `dessert_py` 自定义序列化） | 包含哈希表、质心映射等内部结构 | 完整的 Dessert 索引文件，由 `dessert_py.DocRetrieval.serialize_to_file` 生成。每个 `n_table` 值生成一个独立的索引文件。 | **必需**（检索依赖） |
| `--username`<br>`--dataset`<br>`topk`<br>构建参数 `n_table`<br>检索参数组合 | **检索排名文件**<br>`Result/answer/{dataset}-dessert-top{topk}-n_table_{num_tables}-initial_filter_k_{...}-nprobe_query_{...}-remove_centroid_dupes_{...}-n_thread_{...}.tsv` | TSV 文件，UTF-8 编码，`\t` 分隔 | 每行四列：`qid`, `pid`, `rank`, `score`（score 固定为 1） | 每个查询的 top-k 文档排名结果。`pid` 为文档 ID（字符串 “ID” 前缀被去除，转为整数）。 | **必需**（评估依赖） |
| `--username`<br>`--dataset`<br>构建参数 `n_table` | **构建性能 JSON**<br>`Result/performance/{dataset}-build_index-dessert-n_table_{num_tables}-time.json` | JSON 文件，UTF-8 编码 | 字典：`{"build_index_time_except_centroid(s)": float, "hashes_per_table": int}` | 记录构建索引（不含聚类）的耗时和自动计算的哈希数。 | **可选**（性能分析） |
| `--username`<br>`--dataset`<br>`topk`<br>构建参数 `n_table`<br>检索参数组合 | **检索性能 JSON**<br>`Result/performance/{dataset}-retrieval-dessert-top{topk}-n_table_{num_tables}-initial_filter_k_{...}-nprobe_query_{...}-remove_centroid_dupes_{...}-n_thread_{...}.json` | JSON 文件，UTF-8 编码 | 嵌套字典，包含 `n_query`, `topk`, `build_index`, `retrieval`, `search_time` 等字段 | 记录本次检索的时间性能指标（总耗时、百分位数、各阶段平均耗时、平均精排向量数等）。 | **必需**（性能分析） |
| `--username`<br>`--dataset`<br>`topk`<br>构建参数 `n_table`<br>检索参数组合 | **单查询性能 CSV**（注：代码中注释掉了写入，实际未生成） | CSV（注释掉，未实际输出） | – | 原计划记录每个查询的时间、召回等，但当前版本不输出。 | 非必需（未生成） |

---















## IV. Evaluation Metrics


| 类别 | CSV 字段名 | 单位 | 计算方式 | 详细说明 |
| :--- | :--- | :--- | :--- | :--- |
| **文件标识** | `answer_file` | – | 直接取文件名 | 检索结果 TSV 文件的名称，用于追溯具体实验。 |
| | `performance_file` | – | 根据 `answer_file` 替换后缀和关键字推断得到 | 对应的性能 JSON 文件名，用于关联速度数据。 |
| **方法标识** | `method` | – | 从 `answer_file` 中匹配关键词（plaid / dessert / MUVERA / IGP），均不匹配则为 unknown | 标明本次检索使用的引擎名称。 |
| **基础信息** | `n_query` | – | 直接取自性能 JSON 中的 `n_query` 字段 | 本次检索的查询总数。 |
| | `topk` | – | 直接取自性能 JSON 中的 `topk` 字段 | 检索返回的前 k 个文档数。 |
| **速度/吞吐** | `qps` | 次/秒 | `(1000.0 × n_query) / total_query_time_ms` | 每秒可处理的查询数量（Queries Per Second），反映系统吞吐能力。 |
| | `total_query_time_ms` | 毫秒 | 所有查询的 `retrieval_time_l` 列表元素求和：`sum(retrieval_time_l)` | 所有查询的检索总耗时。 |
| | `average_query_time_ms` | 毫秒 | `total_query_time_ms / n_query` | 每个查询的平均检索耗时。 |
| **构建配置与耗时**<br>（动态列，前缀 `build_`） | `build_index_time` | 秒 | `Indexer.index()` 返回的 `build_index_time` | 索引构建总耗时（含聚类、量化、写入）。 |
| | `build_encode_passage_time` | 秒 | `Indexer.index()` 返回的 `encode_passage_time` | 文档编码耗时（若使用预计算嵌入则为 0）。 |
| | 其他 `build_*` 字段 | 根据配置 | 取自性能 JSON 中 `build_index` 对象内的其他键值 | 可能包含索引构建时的其他参数（如 `nbits` 等）。 |
| **检索参数**<br>（动态列，前缀 `retrieval_`） | `retrieval_ndocs` | 个 | 取自性能 JSON 中 `retrieval.ndocs` | IVF 阶段返回的候选文档数。 |
| | `retrieval_ncells` | 个 | 取自性能 JSON 中 `retrieval.ncells` | IVF 聚类中心数量。 |
| | `retrieval_centroid_score_threshold` | – | 取自性能 JSON 中 `retrieval.centroid_score_threshold` | 过滤阶段使用的质心分数阈值。 |
| | `retrieval_n_thread` | 个 | 取自性能 JSON 中 `retrieval.n_thread` | 检索时使用的 CPU 线程数。 |
| | 其他 `retrieval_*` 字段 | 根据配置 | 取自性能 JSON 中 `retrieval` 对象内的其他键值 | 可能包含其他检索参数（如 `partitions` 等）。 |
| **准确率指标**<br>（每个 k 值生成 4 列） | `recall@{k}` | 比率 [0,1] | 对每个查询：<br> (检索到的前k个文档 ∩ 所有相关文档) / (所有相关文档), 然后对所有查询取**算术平均值**。<br> | 衡量检索到的相关文档覆盖比例。 |
| | `mrr@{k}` | 比率 [0,1] | 对每个查询：<br> 若前k个结果中包含相关文档，取**第一个相关文档排名**的倒数（`1/rank`），否则为 0。<br>然后对所有查询取**算术平均值**。 | 反映第一个相关文档出现的位置。 |
| | `success@{k}` | 比率 [0,1] | 对每个查询：<br> 若前k个结果中**至少有一个**相关文档，则为 1，否则为 0。<br>然后对所有查询取**算术平均值**。 | 表示“至少检索到一个相关文档”的查询占比。 |
| | `ndcg@{k}` | 比率 [0,1] | 对每个查询：<br> 1. 计算 **DCG**：对前k个结果，若文档相关，累加 `1 / log₂(rank + 1)`。<br> 2. 计算 **IDCG**：假设前 `min(相关文档总数, k)` 个结果全部相关时的 DCG 最大值。<br> 3. **NDCG = DCG / IDCG**（若 IDCG = 0 则结果为 0）。<br>然后对所有查询取**算术平均值**。 | 衡量排序质量，越靠前的相关文档贡献越大。 |

---

**补充说明**

- **动态列**：`build_*` 和 `retrieval_*` 开头的列数量不固定，取决于性能 JSON 中实际存储的键。上述表格仅列出了代码中明确会出现的常见字段。
- **k 值**：准确率指标针对 `--k-values` 参数中指定的每个 k 分别计算，默认是 `[10, 100]`，因此默认会生成 `recall@10`、`mrr@10` 等共 8 列（4 个指标 × 2 个 k 值）。
- **所有指标均以“查询”为单位进行平均**，即最终输出的是**宏平均**（每个查询同等权重）。

---
