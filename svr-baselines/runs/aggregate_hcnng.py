import json
import os
import re
import pandas as pd

def extract_hcnng_metrics(base_dir: str, verbose: bool = True):
    """
    遍历 base_dir 下的 maxcalc_* 子目录，提取每个 max_calc 的 BEIR 指标和 QPS，
    生成一个 summary.csv 保存到 base_dir 中。
    
    参数:
        base_dir (str): 包含 maxcalc_* 子目录的根目录（如 ./hcnng/n500_m20）
        verbose (bool): 是否打印进度信息
    """
    rows = []
    # 遍历所有 maxcalc_* 子目录
    for item in os.listdir(base_dir):
        if not item.startswith("maxcalc_"):
            continue
        sub_dir = os.path.join(base_dir, item)
        if not os.path.isdir(sub_dir):
            continue

        # 提取 max_calc 数值
        match = re.search(r"maxcalc_(\d+)", item)
        if not match:
            if verbose:
                print(f"警告: 无法从 {item} 提取 max_calc，跳过")
            continue
        max_calc = int(match.group(1))

        # ---- 读取 JSON 指标 ----
        json_path = os.path.join(sub_dir, "beir_metrics_k100_scored.json")
        if not os.path.isfile(json_path):
            if verbose:
                print(f"警告: {json_path} 不存在，跳过")
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"警告: {json_path} JSON 解析失败 ({e})，跳过")
            continue

        # 提取 K 值（通常为 100）
        k_values = data.get("k_values", [])
        if not k_values:
            if verbose:
                print(f"警告: {json_path} 中无 k_values，跳过")
            continue
        K = k_values[0]

        # ---- 读取 QPS ----
        qps_path = os.path.join(sub_dir, "qps_k100_scored.tsv")
        if not os.path.isfile(qps_path):
            if verbose:
                print(f"警告: {qps_path} 不存在，跳过")
            continue

        try:
            qps_df = pd.read_csv(qps_path, sep='\t')
            if 'QPS' not in qps_df.columns:
                if verbose:
                    print(f"警告: {qps_path} 中缺少 QPS 列，跳过")
                continue
            qps_value = qps_df['QPS'].iloc[0]
            if pd.isna(qps_value):
                if verbose:
                    print(f"警告: {qps_path} 中 QPS 值为空，跳过")
                continue
        except Exception as e:
            if verbose:
                print(f"警告: 读取 {qps_path} 失败 ({e})，跳过")
            continue

        # 组装本行数据
        row = {
            "K": K,
            "max_calc": max_calc,
            "Recall@100": data.get("recall", {}).get("Recall@100"),
            "NDCG@100": data.get("ndcg", {}).get("NDCG@100"),
            "MRR@100": data.get("mrr", {}).get("MRR@100"),
            "MAP@100": data.get("map", {}).get("MAP@100"),
            "P@100": data.get("precision", {}).get("P@100"),
            "Hole@100": data.get("hole", {}).get("Hole@100"),
            "Accuracy@100": data.get("accuracy", {}).get("Accuracy@100"),
            "R_cap@100": data.get("recall_cap", {}).get("R_cap@100"),
            "queries_qrels": data.get("queries_qrels"),
            "queries_results": data.get("queries_results"),
            "QPS": qps_value,
        }
        rows.append(row)

    if not rows:
        if verbose:
            print(f"在 {base_dir} 中未找到任何有效数据，不生成 CSV")
        return

    # 转换为 DataFrame 并按 max_calc 排序
    df = pd.DataFrame(rows).sort_values("max_calc").reset_index(drop=True)

    # 保存到 base_dir 下的 summary.csv
    output_csv = os.path.join(base_dir, "summary.csv")
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"汇总结果已保存至: {output_csv}")


def batch_extract(root_dir: str, verbose: bool = True):
    """
    批量处理 root_dir 下所有 n*_m* 子目录，为每个子目录生成 summary.csv
    
    参数:
        root_dir (str): 包含所有 n*_m* 子目录的父目录（如 ./hcnng）
    """
    for item in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, item)
        if os.path.isdir(sub_path) and re.match(r"n\d+_m\d+", item):
            if verbose:
                print(f"\n处理组合: {item}")
            extract_hcnng_metrics(sub_path, verbose=verbose)


if __name__ == "__main__":
    # 示例用法：直接指定 HCNNG_DIR 进行批处理
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # 默认当前目录，可根据实际情况修改
        target_dir = "/data1/chenyifeng/MultiVector-Backup/svr-baselines/runs/scidocs-single/hcnng"
    
    batch_extract(target_dir)