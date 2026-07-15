import time
import torch
import os
import pathlib
from math import ceil
from torch.utils.cpp_extension import load

from colbert.utils.utils import flatten, print_message
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided
from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration
from .index_loader import IndexLoader
from colbert.modeling.colbert import colbert_score, colbert_score_packed, colbert_score_reduce

class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(self, index_path, use_gpu=True):
        super().__init__(index_path=index_path, use_gpu=use_gpu)
        IndexScorer.try_load_torch_extensions(use_gpu)
        self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens)

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return
        
        print_message(f"Loading filter_pids_cpp extension...")
        filter_pids_cpp = load(
            name="filter_pids_cpp",
            sources=[os.path.join(pathlib.Path(__file__).parent.resolve(), "filter_pids.cpp")],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.filter_pids = filter_pids_cpp.filter_pids_cpp

        print_message(f"Loading decompress_residuals_cpp extension...")
        decompress_residuals_cpp = load(
            name="decompress_residuals_cpp",
            sources=[os.path.join(pathlib.Path(__file__).parent.resolve(), "decompress_residuals.cpp")],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.decompress_residuals = decompress_residuals_cpp.decompress_residuals_cpp
        cls.loaded_extensions = True

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        return self.embeddings_strided.lookup_eids(embedding_ids, codes=codes, out_device=out_device)

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def retrieve(self, config, Q):
        Q = Q[:, :config.query_maxlen]
        embedding_ids, centroid_scores = self.generate_candidates(config, Q)
        return embedding_ids, centroid_scores

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(self.emb2pid[embedding_ids.long()].cuda(), sorted=False)
        return all_pids

    def rank(self, config, Q, filter_fn=None):
        with torch.inference_mode():
            start_ivf_time = time.time_ns()
            pids, centroid_scores = self.retrieve(config, Q)
            end_ivf_time = time.time_ns()
            ivf_time = (end_ivf_time - start_ivf_time) * 1e-6

            if filter_fn is not None:
                pids = filter_fn(pids)

            scores, pids, filter_time, refine_time, n_refine_ivf, n_refine_filter, n_vec_score_refine = \
                self.score_pids(config, Q, pids, centroid_scores)

            scores_sorter = scores.sort(descending=True)
            indices = scores_sorter.indices.cpu()
            pids, scores = pids[indices].tolist(), scores_sorter.values.tolist()

            return pids, scores, ivf_time, filter_time, refine_time, n_refine_ivf, n_refine_filter, n_vec_score_refine

    def score_pids(self, config, Q, pids, centroid_scores):
        start_filter_time = time.time_ns()
        
        # Batch size 
        batch_size = 2 ** 17

        if self.use_gpu:
            centroid_scores = centroid_scores.cuda()

        # Pre-calc mask
        t_mask_start = time.time_ns()
        idx = centroid_scores.max(-1).values >= config.centroid_score_threshold
        t_mask_end = time.time_ns()
        
        n_refine_ivf = len(pids)
        s2_total_ns = 0
        s3_total_ns = 0

        if self.use_gpu:
            print(f"\n{'='*20} [Query Analysis] {'='*20}")
            print(f"Candidates: {len(pids)}, Pre-calc Mask: {(t_mask_end - t_mask_start)/1e6:.3f} ms")

            approx_scores = []

            # ==========================================
            # Stage 2: Pruned Interaction (Batched)
            # ==========================================
            s2_start = time.time_ns()
            
            # --- 初始化 Stage 2 累加器 ---
            sum_s2_a = 0.0 # Gather Codes
            sum_s2_b = 0.0 # Apply Mask
            sum_s2_c = 0.0 # Strided Meta
            sum_s2_d = 0.0 # Expand/Pad
            sum_s2_e = 0.0 # Sum Lengths
            sum_s2_f = 0.0 # Gather Scores
            sum_s2_g = 0.0 # Score Meta
            sum_s2_h = 0.0 # Score Expand
            sum_s2_i = 0.0 # Reduce
            
            num_batches = ceil(len(pids) / batch_size)
            
            for i in range(num_batches):
                pids_ = pids[i * batch_size: (i + 1) * batch_size]
                t_check = time.time_ns()

                # 1. Gather Codes
                codes_packed, codes_lengths = self.embeddings_strided.lookup_codes(pids_)
                t_gather = time.time_ns()
                sum_s2_a += (t_gather - t_check)

                # 2. Apply Mask
                idx_ = idx[codes_packed.long()]
                t_mask_idx = time.time_ns()
                sum_s2_b += (t_mask_idx - t_gather)

                # 3. Create StridedTensor Meta
                pruned_codes_strided = StridedTensor(idx_, codes_lengths, use_gpu=self.use_gpu)
                t_strided = time.time_ns()
                sum_s2_c += (t_strided - t_mask_idx)

                # 4. Expand/Pad
                pruned_codes_padded, pruned_codes_mask = pruned_codes_strided.as_padded_tensor()
                t_expand = time.time_ns()
                sum_s2_d += (t_expand - t_strided)

                # 5. Sum Lengths
                pruned_codes_lengths = (pruned_codes_padded * pruned_codes_mask).sum(dim=1)
                t_sum = time.time_ns()
                sum_s2_e += (t_sum - t_expand)

                # 6. Gather Scores
                codes_packed_ = codes_packed[idx_]
                approx_scores_ = centroid_scores[codes_packed_.long()]
                t_score_gather = time.time_ns()
                sum_s2_f += (t_score_gather - t_sum)

                if approx_scores_.shape[0] == 0:
                    approx_scores.append(torch.zeros((len(pids_),), dtype=approx_scores_.dtype).cuda())
                    continue
                
                # 7. Create Score Strided
                approx_scores_strided = StridedTensor(approx_scores_, pruned_codes_lengths, use_gpu=self.use_gpu)
                t_score_strided = time.time_ns()
                sum_s2_g += (t_score_strided - t_score_gather)

                # 8. Score Expand
                approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
                t_score_expand = time.time_ns()
                sum_s2_h += (t_score_expand - t_score_strided)

                # 9. Reduce (MaxSim)
                approx_scores_ = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
                t_reduce = time.time_ns()
                sum_s2_i += (t_reduce - t_score_expand)

                approx_scores.append(approx_scores_)

            # Merge results
            t_merge_start = time.time_ns()
            approx_scores = torch.cat(approx_scores, dim=0)
            t_merge_end = time.time_ns()
            
            # Selection
            t_topk_start = time.time_ns()
            if config.ndocs < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=config.ndocs).indices]
            t_topk_end = time.time_ns()
            
            s2_end = time.time_ns()
            s2_total_ns = s2_end - s2_start
            
            # --- Print Stage 2 Breakdown (Aggregated) ---
            print(f"  --- [Stage 2 Breakdown] (Total Batches: {num_batches}) ---")
            print(f"    #L126 : {sum_s2_a/1e6:.3f} ms")
            print(f"    #L127   : {sum_s2_b/1e6:.3f} ms")
            print(f"    #L128 : {sum_s2_c/1e6:.3f} ms")
            print(f"    #L129   : {sum_s2_d/1e6:.3f} ms")
            print(f"    #L130  : {sum_s2_e/1e6:.3f} ms")
            print(f"    #L131-132 : {sum_s2_f/1e6:.3f} ms")
            print(f"    #L136   : {sum_s2_g/1e6:.3f} ms")
            print(f"    #L137 : {sum_s2_h/1e6:.3f} ms")
            print(f"    #L138: {sum_s2_i/1e6:.3f} ms")
            print(f"    #L140  : {(t_merge_end - t_merge_start)/1e6:.3f} ms")
            print(f"    #L142-143 : {(t_topk_end - t_topk_start)/1e6:.3f} ms")

            # ==========================================
            # Stage 3: Full Interaction
            # ==========================================
            s3_start = time.time_ns()
            # 通常 Stage 3 是一次性运行，直接打印即可
            t_check = time.time_ns()
            
            # 1. Gather
            codes_packed, codes_lengths = self.embeddings_strided.lookup_codes(pids)
            t_gather = time.time_ns()
            
            # 2. Lookup Scores
            approx_scores = centroid_scores[codes_packed.long()]
            t_score_lookup = time.time_ns()
            
            # 3. Strided Meta
            approx_scores_strided = StridedTensor(approx_scores, codes_lengths, use_gpu=self.use_gpu)
            t_strided = time.time_ns()
            
            # 4. Expand
            approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
            t_expand = time.time_ns()
            
            # 5. Reduce
            approx_scores = colbert_score_reduce(approx_scores_padded, approx_scores_mask, config)
            t_reduce = time.time_ns()

            # Selection
            if config.ndocs // 4 < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=(config.ndocs // 4)).indices]
            
            s3_end = time.time_ns()
            s3_total_ns = s3_end - s3_start

            print(f"  --- [Stage 3 Breakdown] ({len(pids)} docs) ---")
            print(f"    #L146 : {(t_gather - t_check)/1e6:.3f} ms")
            print(f"    #L147 : {(t_score_lookup - t_gather)/1e6:.3f} ms")
            print(f"    #L148 : {(t_strided - t_score_lookup)/1e6:.3f} ms")
            print(f"    #L149 : {(t_expand - t_strided)/1e6:.3f} ms")
            print(f"    #L150 : {(t_reduce - t_expand)/1e6:.3f} ms")

            # === Summary ===
            print(f"  >>> [Summary] Stage 2 Total: {s2_total_ns/1e6:.3f} ms | Stage 3 Total: {s3_total_ns/1e6:.3f} ms")
            
            n_vec_score_refine = 0

        else:
            # CPU Path
            pids, n_vec_score_refine = IndexScorer.filter_pids(
                pids, centroid_scores, self.embeddings.codes, self.doclens,
                self.embeddings_strided.codes_strided.offsets, idx, config.ndocs
            )
        
        n_refine_filter = len(pids)
        end_filter_time = time.time_ns()
        filter_time = (end_filter_time - start_filter_time) * 1e-6

        start_refine_time = time.time_ns()
        
        # Stage 4
        if self.use_gpu:
            D_packed, D_mask = self.lookup_pids(pids)
        else:
            D_packed = IndexScorer.decompress_residuals(
                pids, self.doclens, self.embeddings_strided.codes_strided.offsets,
                self.codec.bucket_weights, self.codec.reversed_bit_map,
                self.codec.decompression_lookup_table, self.embeddings.residuals,
                self.embeddings.codes, self.codec.centroids, self.codec.dim, self.codec.nbits
            )
            D_packed = torch.nn.functional.normalize(D_packed.to(torch.float32), p=2, dim=-1)
            D_mask = self.doclens[pids.long()]

        if Q.size(0) == 1:
            scores = colbert_score_packed(Q, D_packed, D_mask, config)
            end_refine_time = time.time_ns()
            refine_time = (end_refine_time - start_refine_time) * 1e-6
            return scores, pids, filter_time, refine_time, n_refine_ivf, n_refine_filter, n_vec_score_refine

        D_strided = StridedTensor(D_packed, D_mask, use_gpu=self.use_gpu)
        D_padded, D_lengths = D_strided.as_padded_tensor()

        scores = colbert_score(Q, D_padded, D_lengths, config)
        end_refine_time = time.time_ns()
        refine_time = (end_refine_time - start_refine_time) * 1e-6
        return scores, pids, filter_time, refine_time, n_refine_ivf, n_refine_filter, n_vec_score_refine