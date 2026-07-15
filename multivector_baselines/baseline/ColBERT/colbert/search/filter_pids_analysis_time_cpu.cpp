// #include <pthread.h>
// #include <torch/extension.h>

// #include <algorithm>
// #include <chrono>
// #include <numeric>
// #include <utility>
// #include <iostream>
// #include <iomanip>
// #include <vector>

// // 宏定义方便计时
// #define NOW std::chrono::high_resolution_clock::now()
// #define DUR_NS(start, end) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()

// typedef struct maxsim_args
// {
//     int tid;
//     int nthreads;

//     int ncentroids;
//     int nquery_vectors;
//     int npids;

//     int* pids;
//     float* centroid_scores;
//     int* codes;
//     int64_t* doclens;
//     int64_t* offsets;
//     bool* idx;

//     size_t n_vec_score_refine;

//     std::priority_queue<std::pair<float, int>> approx_scores;

//     // [计时累加器] (纳秒级，避免精度损失)
//     long long t_total_loop_ns = 0; // 整个循环的总时间
//     long long t_core_calc_ns = 0;  // 核心计算(查表+MaxSim)时间
//     long long t_queue_push_ns = 0; // 结果入队时间
// } maxsim_args_t;

// void* maxsim(void* args)
// {
//     maxsim_args_t* maxsim_args = (maxsim_args_t*)args;

//     float per_doc_approx_scores[maxsim_args->nquery_vectors];
//     for (int k = 0; k < maxsim_args->nquery_vectors; k++)
//     {
//         per_doc_approx_scores[k] = -9999;
//     }

//     int ndocs_per_thread =
//         (int)std::ceil(((float)maxsim_args->npids) / maxsim_args->nthreads);
//     int start = maxsim_args->tid * ndocs_per_thread;
//     int end =
//         std::min((maxsim_args->tid + 1) * ndocs_per_thread, maxsim_args->npids);

//     std::unordered_set<int> seen_codes;

//     // === [Line A] Loop Start ===
//     auto t_loop_start = NOW;

//     for (int i = start; i < end; i++)
//     {
//         auto pid = maxsim_args->pids[i];
        
//         // [隐式耗时] Token遍历与Mask检查 (Traverse & Prune)
//         for (int j = 0; j < maxsim_args->doclens[pid]; j++)
//         {
//             auto code = maxsim_args->codes[maxsim_args->offsets[pid] + j];
//             assert(code < maxsim_args->ncentroids);
            
//             if (maxsim_args->idx[code] &&
//                 seen_codes.find(code) == seen_codes.end())
//             {
//                 // === [Line B] Core Calculation Start ===
//                 // 这是一个极高频的内部循环，为了性能，我们尽量减少这里的计时频率
//                 // 但为了统计精确，我们必须在这里打点
//                 auto t_calc_start = NOW;

//                 for (int k = 0; k < maxsim_args->nquery_vectors; k++)
//                 {
//                     per_doc_approx_scores[k] =
//                         std::max(per_doc_approx_scores[k],
//                                  maxsim_args->centroid_scores
//                                  [code * maxsim_args->nquery_vectors + k]);
//                 }
//                 maxsim_args->n_vec_score_refine += maxsim_args->nquery_vectors;
                
//                 auto t_calc_end = NOW;
//                 maxsim_args->t_core_calc_ns += DUR_NS(t_calc_start, t_calc_end);
//                 // === [Line B] Core Calculation End ===

//                 seen_codes.insert(code);
//             }
//         }
        
//         float score = 0;
//         for (int k = 0; k < maxsim_args->nquery_vectors; k++)
//         {
//             score += per_doc_approx_scores[k];
//             per_doc_approx_scores[k] = -9999;
//         }
//         seen_codes.clear();

//         // === [Line C] Queue Push Start ===
//         auto t_push_start = NOW;
//         maxsim_args->approx_scores.push(std::make_pair(score, pid));
//         auto t_push_end = NOW;
//         maxsim_args->t_queue_push_ns += DUR_NS(t_push_start, t_push_end);
//         // === [Line C] Queue Push End ===
//     }

//     auto t_loop_end = NOW;
//     maxsim_args->t_total_loop_ns += DUR_NS(t_loop_start, t_loop_end);
//     // === [Line A] Loop End ===

//     return NULL;
// }

// // 辅助函数：打印统计信息
// void print_breakdown(const std::string& stage_name, int nthreads, maxsim_args_t* args, 
//                      long long t_spawn, long long t_join, long long t_merge, int num_docs) {
    
//     long long sum_total_loop = 0;
//     long long sum_core_calc = 0;
//     long long sum_queue_push = 0;

//     // 汇总所有线程的时间
//     for(int i=0; i<nthreads; ++i) {
//         sum_total_loop += args[i].t_total_loop_ns;
//         sum_core_calc += args[i].t_core_calc_ns;
//         sum_queue_push += args[i].t_queue_push_ns;
//     }

//     // 计算平均值（模拟单线程视角或墙上时间贡献）
//     // 注意：因为是并行执行，Threads Total Sum 会大于 Wall Time。
//     // 为了对比瓶颈，我们看 Sum 或者 Max 都可以。这里展示 Sum (总CPU时间) 更能反应工作量。
    
//     long long sum_token_nav = sum_total_loop - sum_core_calc - sum_queue_push;
//     long long total_cpu_time = sum_total_loop + t_spawn + t_join + t_merge; 
    
//     // 转换为毫秒
//     double ms_token = sum_token_nav / 1e6;
//     double ms_calc = sum_core_calc / 1e6;
//     double ms_push = sum_queue_push / 1e6;
//     double ms_spawn = t_spawn / 1e6;
//     double ms_join = t_join / 1e6;
//     double ms_merge = t_merge / 1e6;
//     double ms_total = total_cpu_time / 1e6;

//     std::cout << "\n  --- [" << stage_name << "] Breakdown (Docs: " << num_docs << ") ---" << std::endl;
//     std::cout << std::fixed << std::setprecision(3);
//     std::cout << "    1. Thread Spawn      : " << std::setw(8) << ms_spawn << " ms" << std::endl;
//     std::cout << "    2. Token Nav & Mask  : " << std::setw(8) << ms_token << " ms (CPU Mem Access)" << std::endl;
//     std::cout << "    3. Core Calc (Score) : " << std::setw(8) << ms_calc  << " ms (Inner Loop)" << std::endl;
//     std::cout << "    4. Queue Push        : " << std::setw(8) << ms_push  << " ms" << std::endl;
//     std::cout << "    5. Thread Join (Wait): " << std::setw(8) << ms_join  << " ms" << std::endl;
//     std::cout << "    6. Merge Results     : " << std::setw(8) << ms_merge << " ms" << std::endl;
//     std::cout << "    ----------------------------------------" << std::endl;
//     std::cout << "    >> TOTAL CPU TIME    : " << std::setw(8) << ms_total << " ms" << std::endl;
// }

// void filter_pids_helper(int ncentroids, int nquery_vectors, int npids,
//                         int* pids, float* centroid_scores, int* codes,
//                         int64_t* doclens, int64_t* offsets, bool* idx,
//                         int nfiltered_docs, int* filtered_pids, size_t& n_vec_score_refine,
//                         const std::string& stage_label)
// {
//     auto nthreads = at::get_num_threads();

//     pthread_t threads[nthreads];
//     maxsim_args_t args[nthreads];

//     auto t0 = NOW; // Spawn Start

//     for (int i = 0; i < nthreads; i++)
//     {
//         args[i].tid = i;
//         args[i].nthreads = nthreads;
//         args[i].ncentroids = ncentroids;
//         args[i].nquery_vectors = nquery_vectors;
//         args[i].npids = npids;
//         args[i].pids = pids;
//         args[i].centroid_scores = centroid_scores;
//         args[i].codes = codes;
//         args[i].doclens = doclens;
//         args[i].offsets = offsets;
//         args[i].idx = idx;
//         args[i].n_vec_score_refine = 0;
//         args[i].approx_scores = std::priority_queue<std::pair<float, int>>();

//         // Reset counters
//         args[i].t_total_loop_ns = 0;
//         args[i].t_core_calc_ns = 0;
//         args[i].t_queue_push_ns = 0;

//         int rc = pthread_create(&threads[i], NULL, maxsim, (void*)&args[i]);
//         if (rc)
//         {
//             fprintf(stderr, "Unable to create thread %d: %d\n", i, rc);
//             std::exit(1);
//         }
//     }

//     auto t1 = NOW; // Spawn End / Join Start

//     for (int i = 0; i < nthreads; i++)
//     {
//         pthread_join(threads[i], NULL);
//     }

//     auto t2 = NOW; // Join End / Merge Start

//     std::priority_queue<std::pair<float, int>> global_approx_scores;
//     n_vec_score_refine = 0;
//     for (int i = 0; i < nthreads; i++)
//     {
//         for (int j = 0; j < nfiltered_docs; j++)
//         {
//             if (args[i].approx_scores.empty())
//             {
//                 break;
//             }
//             global_approx_scores.push(args[i].approx_scores.top());
//             args[i].approx_scores.pop();
//         }
//         n_vec_score_refine += args[i].n_vec_score_refine;
//     }

//     for (int i = 0; i < nfiltered_docs; i++)
//     {
//         std::pair<float, int> score_and_pid = global_approx_scores.top();
//         filtered_pids[i] = score_and_pid.second;
//         global_approx_scores.pop();
//     }

//     auto t3 = NOW; // Merge End

//     // 打印统计
//     long long t_spawn = DUR_NS(t0, t1);
//     long long t_join = DUR_NS(t1, t2);
//     long long t_merge = DUR_NS(t2, t3);

//     print_breakdown(stage_label, nthreads, args, t_spawn, t_join, t_merge, npids);
// }

// std::pair<torch::Tensor, size_t> filter_pids(const torch::Tensor pids,
//                                              const torch::Tensor centroid_scores,
//                                              const torch::Tensor codes,
//                                              const torch::Tensor doclens,
//                                              const torch::Tensor offsets, const torch::Tensor idx,
//                                              int nfiltered_docs)
// {
//     auto ncentroids = centroid_scores.size(0);
//     auto nquery_vectors = centroid_scores.size(1);
//     auto npids = pids.size(0);

//     auto pids_a = pids.data_ptr<int>();
//     auto centroid_scores_a = centroid_scores.data_ptr<float>();
//     auto codes_a = codes.data_ptr<int>();
//     auto doclens_a = doclens.data_ptr<int64_t>();
//     auto offsets_a = offsets.data_ptr<int64_t>();
//     auto idx_a = idx.data_ptr<bool>();

//     size_t first_n_vec_score_refine = 0;
//     int filtered_pids[nfiltered_docs];
    
//     std::cout << "\n================ CPU Query Profile ================" << std::endl;

//     // === Stage 2 ===
//     filter_pids_helper(ncentroids, nquery_vectors, npids, pids_a,
//                        centroid_scores_a, codes_a, doclens_a, offsets_a, idx_a,
//                        nfiltered_docs, filtered_pids, first_n_vec_score_refine, "Stage 2: Pruned");

//     int nfinal_filtered_docs = (int)(nfiltered_docs / 4);
//     int final_filtered_pids[nfinal_filtered_docs];
//     bool ones[ncentroids];
//     for (int i = 0; i < ncentroids; i++)
//     {
//         ones[i] = true;
//     }
    
//     size_t second_n_vec_score_refine = 0;
    
//     // === Stage 3 ===
//     filter_pids_helper(ncentroids, nquery_vectors, nfiltered_docs,
//                        filtered_pids, centroid_scores_a, codes_a, doclens_a,
//                        offsets_a, ones, nfinal_filtered_docs,
//                        final_filtered_pids, second_n_vec_score_refine, "Stage 3: Full");

//     size_t n_vec_score_refine = first_n_vec_score_refine + second_n_vec_score_refine;

//     auto options =
//         torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
//     return std::make_pair(torch::from_blob(final_filtered_pids, {nfinal_filtered_docs},
//                                            options)
//                           .clone(), n_vec_score_refine);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
//     m.def("filter_pids_cpp", &filter_pids, "Filter pids");
// }