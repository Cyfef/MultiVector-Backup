# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import pytrec_eval
import logging
from typing import List, Dict, Tuple
import sys
import csv
import collections

def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        if ignore_identical_ids:
            logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        # -------------------- 计算 NDCG, MAP, Recall, Precision (保持原逻辑不变) --------------------
        # 仅为了原指标截断到最大的 k，因为原指标内部有 cut 参数
        max_k = max(k_values)
        truncated_results = {}
        for qid, docs in results.items():
            sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:max_k]
            truncated_results[qid] = dict(sorted_docs)
        
        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(truncated_results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)

        # -------------------- 新增：专门计算 MRR@1, @10, @100 (因为 recip_rank 不支持 cut) --------------------
        mrr = {}
        for k in k_values:
            # 1. 针对当前 k，将每个 query 的检索结果截断到前 k 个文档
            truncated_for_k = {}
            for qid, docs in results.items():
                sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]
                truncated_for_k[qid] = dict(sorted_docs)
            
            # 2. 单独用 recip_rank 评估截断后的结果
            evaluator_mrr = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
            scores_mrr = evaluator_mrr.evaluate(truncated_for_k)
            
            # 3. 累加平均
            mrr_sum = 0.0
            for qid in scores_mrr.keys():
                mrr_sum += scores_mrr[qid]["recip_rank"]
            mrr[f"MRR@{k}"] = round(mrr_sum / len(scores_mrr), 5)

        # -------------------- 打印所有指标 --------------------
        for metric in [ndcg, _map, recall, precision]:
            print("\n")
            for key in metric.keys():
                print("{}: {:.4f}".format(key, metric[key]))
        
        print("\n")
        for key in mrr.keys():
            print("{}: {:.4f}".format(key, mrr[key]))
        
        # 返回所有指标（多了一个 mrr）
        return ndcg, _map, recall, precision, mrr

def load_qrels(path):
    qrels = {}
    reader = csv.reader(open(path, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    for row in reader:
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: int(score)}
        else:
            qrels[query_id][corpus_id] = int(score)
    return qrels

def load_results(path):
    with open(path) as f:
        lines = f.readlines()
    results = collections.defaultdict(dict)
    for line in lines:
        qid, _, doc_id, _, score, _ = line.strip().split(" ")
        results[qid][doc_id] = float(score)
    return results

def main():
    qrels = load_qrels(sys.argv[1])
    results = load_results(sys.argv[2])
    # 这里返回 5 个值，如果你有接收变量的地方请改成 5 个
    ndcg, _map, recall, precision, mrr = evaluate(qrels, results, [1, 10, 100])
    # 如果你只想看打印，直接调用 evaluate 而不接收返回值也可以

if __name__ == "__main__":
    main()