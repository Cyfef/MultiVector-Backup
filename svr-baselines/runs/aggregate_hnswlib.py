import json
import os
import re
import pandas as pd

def extract_hnsw_metrics(base_dir: str, logs_dir: str):
    """
    遍历 HNSW 结果目录（M*_efc* 子目录），在每个组合下提取所有 ef_search 的指标和 QPS，
    生成汇总 CSV 文件保存在对应组合目录下（文件名：summary.csv）。
    """
    # ---------- 1. 检查路径 ----------
    if not os.path.isdir(base_dir):
        print(f"错误: base_dir '{base_dir}' 不存在")
        return
    if not os.path.isdir(logs_dir):
        print(f"错误: logs_dir '{logs_dir}' 不存在")
        return

    combo_pattern = re.compile(r'^M(\d+)_efc(\d+)$')
    total_combos = 0
    success_combos = 0

    for combo_name in os.listdir(base_dir):
        combo_path = os.path.join(base_dir, combo_name)
        if not os.path.isdir(combo_path):
            continue

        match = combo_pattern.match(combo_name)
        if not match:
            continue

        total_combos += 1
        M_val = int(match.group(1))
        efc_val = int(match.group(2))
        print(f"正在处理组合: {combo_name} (M={M_val}, efc={efc_val})")

        rows = []
        skipped_no_json = 0
        skipped_no_qps = 0
        skipped_other = 0

        # 遍历组合下的所有 ef_* 子目录
        ef_dirs = [d for d in os.listdir(combo_path) if os.path.isdir(os.path.join(combo_path, d)) and d.startswith("ef_")]
        if not ef_dirs:
            print(f"  警告: 组合 {combo_name} 下没有 ef_* 子目录，跳过")
            continue

        for ef_item in ef_dirs:
            ef_path = os.path.join(combo_path, ef_item)
            ef_match = re.search(r'ef_(\d+)', ef_item)
            if not ef_match:
                continue
            ef_search = int(ef_match.group(1))

            # ----- 读取 BEIR 指标 JSON -----
            json_path = os.path.join(ef_path, "beir_metrics_k100_scored.json")
            if not os.path.isfile(json_path):
                print(f"  警告: 缺失 {json_path}，跳过")
                skipped_no_json += 1
                continue

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  警告: 读取 {json_path} 失败 ({e})，跳过")
                skipped_other += 1
                continue

            k_values = data.get("k_values", [])
            if not k_values:
                print(f"  警告: {json_path} 中无 k_values，跳过")
                skipped_other += 1
                continue
            K = k_values[0]

            # 提取指标
            row = {
                "K": K,
                "ef_search": ef_search,
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
            }

            # ----- 从对应日志文件中读取 QPS -----
            log_file = os.path.join(logs_dir, f"hnswlib_M{M_val}_efc{efc_val}_ef{ef_search}.log")
            if not os.path.isfile(log_file):
                print(f"  警告: 缺失日志 {log_file}，跳过")
                skipped_no_qps += 1
                continue

            try:
                with open(log_file, "r") as f:
                    content = f.read()
                qps_match = re.search(r'\[QPS\]\s+([\d.]+)', content)
                if not qps_match:
                    print(f"  警告: 在 {log_file} 中未找到 [QPS] 行，跳过")
                    skipped_no_qps += 1
                    continue
                qps_value = float(qps_match.group(1))
                row["QPS"] = qps_value
            except Exception as e:
                print(f"  警告: 读取 {log_file} 失败 ({e})，跳过")
                skipped_other += 1
                continue

            rows.append(row)

        if not rows:
            print(f"  组合 {combo_name} 无有效数据（json缺失:{skipped_no_json}, qps缺失:{skipped_no_qps}, 其他:{skipped_other}），跳过")
            continue

        # 按 ef_search 升序排列
        df = pd.DataFrame(rows).sort_values("ef_search").reset_index(drop=True)

        # 保存到组合目录下
        output_csv = os.path.join(combo_path, "summary.csv")
        df.to_csv(output_csv, index=False)
        print(f"  已保存: {output_csv} (包含 {len(rows)} 个 ef_search 值)")
        success_combos += 1

    print(f"\n处理完成: 共发现 {total_combos} 个组合，成功生成 {success_combos} 个 CSV。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="汇总 HNSW 各组合的搜索结果")
    parser.add_argument("--base-dir", required=True, help="包含 M*_efc* 子目录的根目录")
    parser.add_argument("--logs-dir", required=True, help="日志文件所在目录")
    args = parser.parse_args()
    extract_hnsw_metrics(args.base_dir, args.logs_dir)