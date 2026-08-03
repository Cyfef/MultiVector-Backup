// Binhan Yang 2025-09-04 version
#define ENABLE_FIVECS
#ifndef BYTE_MULTI_VECTOR_DATA
#define BYTE_MULTI_VECTOR_DATA

#include <cstddef>
#include <utility>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cassert>
#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>
#include <faiss/IndexHNSW.h>
#include <cmath>

const auto metric = faiss::MetricType::METRIC_INNER_PRODUCT;

using vdata_t = float;

// ----------------------------------------------------
// 1. 全局 GAMMA 常量设定与归一化乘子
// ----------------------------------------------------
const int GAMMA = 2; 
const float INV_GAMMA = 1.0f / GAMMA; 


struct centroidIndex {
    int n_centroid;
    std::unique_ptr<float[]> centroids;
    std::unique_ptr<faiss::idx_t[]> labels;

    centroidIndex(int n_centroid, int n_doc, int dim) : 
        n_centroid(n_centroid),
        centroids(std::make_unique<float[]>(n_centroid * dim)),
        labels(std::make_unique<faiss::idx_t[]>(n_doc)) {}
};

struct docIndex {
    centroidIndex* _c_index;
    faiss::Index* _knn_index;
};

struct MultiVectorDoc {
    size_t doclen;
    size_t dim;
    docIndex index;
};

const auto MULTIVECTORDOCALIGNSIZE = std::max(alignof(MultiVectorDoc), alignof(vdata_t));

inline size_t _size_doc(size_t doclen, size_t dim) {    //  ...
    return sizeof(MultiVectorDoc) + doclen * dim * sizeof(vdata_t);
}

inline MultiVectorDoc* _make_doc(size_t doclen_, size_t dim_) {
    size_t size_ = _size_doc(doclen_, dim_);
    void *mem = ::operator new(size_, std::align_val_t(MULTIVECTORDOCALIGNSIZE));
    MultiVectorDoc* ptr = static_cast<MultiVectorDoc*>(mem);
    ptr -> doclen = doclen_; ptr -> dim = dim_;
    ptr -> index = {};
    return ptr;
}

inline vdata_t* _data_doc(MultiVectorDoc *ptr) {
    return reinterpret_cast<vdata_t*>(ptr + 1);
}

inline const vdata_t* _data_doc(const MultiVectorDoc *ptr) {
    return reinterpret_cast<const vdata_t*>(ptr + 1);
}

inline vdata_t* _veci_doc(MultiVectorDoc *ptr, int i) {
    return _data_doc(ptr) + i * ptr -> dim;
}

inline void _delete_doc(MultiVectorDoc* docptr_) {
    if(!docptr_) return ;
    size_t mem_size = _size_doc(docptr_ -> doclen, docptr_ -> dim);
    ::operator delete(docptr_, std::align_val_t(MULTIVECTORDOCALIGNSIZE));
}

inline void _copy_array(MultiVectorDoc *doc, vdata_t *arr) {
    memcpy(_data_doc(doc), arr, sizeof(vdata_t) * doc -> doclen * doc -> dim);
}

inline centroidIndex* centroidIndexing(MultiVectorDoc *doc, int niter = 25, int nredo = 5) {
    faiss::ClusteringParameters cp;
    cp.niter = niter;
    cp.nredo = nredo;
    cp.spherical = true;
    cp.min_points_per_centroid = 1; 

    const int k = std::max(1, (int)sqrt(doc -> doclen));
    
    faiss::Clustering clus(doc -> dim, k, cp);
    faiss::IndexFlatIP quant(doc -> dim);
    clus.train(doc -> doclen, _data_doc(doc), quant);

    centroidIndex *ret = new centroidIndex(k, doc -> doclen, doc -> dim);
    std::copy(clus.centroids.begin(), clus.centroids.end(), ret -> centroids.get());
    quant.assign(doc -> doclen, _data_doc(doc), ret -> labels.get(), 1);

    return ret;
}

inline faiss::Index* knnIndexing(MultiVectorDoc *doc) {
    static_assert(std::is_same<vdata_t, float>::value,
                  "FAISS expects float storage; please convert if vdata_t != float");

    const int d = static_cast<int>(doc->dim);
    const int64_t n = static_cast<int64_t>(doc->doclen);

    faiss::Index *idx;
    if (metric == faiss::MetricType::METRIC_INNER_PRODUCT) {
        idx = new faiss::IndexFlatIP(d);
    } else if (metric == faiss::MetricType::METRIC_L2) {
        idx = new faiss::IndexFlatL2(d);
    } else {
        throw std::invalid_argument("Unsupported metric for IndexFlat");
    }
    idx->add(n, _data_doc(doc));
    return idx;
}

inline faiss::Index* appKnnIndexing(MultiVectorDoc *doc) {
    static_assert(std::is_same<vdata_t, float>::value,
                  "FAISS expects float storage; please convert if vdata_t != float");

    const int d = static_cast<int>(doc->dim);
    const int64_t n = static_cast<int64_t>(doc->doclen);

    faiss::IndexHNSWFlat *idx;
    if (metric == faiss::MetricType::METRIC_INNER_PRODUCT) {
        idx = new faiss::IndexHNSWFlat(doc -> dim, 32, faiss::MetricType::METRIC_INNER_PRODUCT);
        idx -> hnsw.efConstruction = 512;
        idx -> hnsw.efSearch = 64;
    } else if (metric == faiss::MetricType::METRIC_L2) {
        idx = new faiss::IndexHNSWFlat(doc -> dim, 16, faiss::MetricType::METRIC_L2);
        idx -> hnsw.efConstruction = 512;
        idx -> hnsw.efSearch = 64;
    } else {
        throw std::invalid_argument("Unsupported metric for IndexFlat");
    }
    idx->add(n, _data_doc(doc));
    return idx;
}

void _build_index(MultiVectorDoc *doc) {
    doc -> index._c_index = centroidIndexing(doc);
    doc -> index._knn_index = knnIndexing(doc);
}

static float
MaxSim(MultiVectorDoc *q, MultiVectorDoc *d) {
    faiss::Index *dIndex = d -> index._knn_index;
    if(dIndex == nullptr) dIndex = d -> index._knn_index = knnIndexing(d);
    
    int n_q = q -> doclen;
    int k = std::min((int)d -> doclen, GAMMA);
    
    float *scores = new float[n_q * k];
    faiss::idx_t *best = new faiss::idx_t[n_q * k];
    
    dIndex -> search(n_q, _data_doc(q), k, scores, best);
    float score = std::accumulate(scores, scores + n_q * k, 0.0f) * INV_GAMMA;
    
    delete[] scores;
    delete[] best;
    return score;
}

inline void generate_candidates(size_t n_centroid, float *centroids, const MultiVectorDoc *doc, faiss::Index** &candidates) {
    const int beta = std::min((int)doc -> doclen, std::max(GAMMA, (int)sqrt(doc -> doclen)));

    float *dis = new float[n_centroid * beta];
    faiss::idx_t *idx = new faiss::idx_t[n_centroid * beta];
    faiss::Index* knnIndex = doc -> index._knn_index;
    knnIndex -> search(n_centroid, centroids, beta, dis, idx);
    delete[] dis;

    candidates = new faiss::Index*[n_centroid];

    for(int i = 0; i < n_centroid; ++ i) {
        candidates[i] = new faiss::IndexFlatIP(doc -> dim);
        for(int j = 0; j < beta; ++ j) {
            int candidate_index = idx[i * beta + j];
            if(candidate_index == -1) break;
            candidates[i] -> add(1, _data_doc(doc) + candidate_index * doc -> dim);
        }
    }

    delete[] idx;
}

inline float reranking(const MultiVectorDoc *q, faiss::Index **candidates) {
    const int n_q = q -> doclen;
    const int dim = q -> dim;
    const faiss::idx_t *labels = q -> index._c_index -> labels.get();

    float *scores = new float[n_q * GAMMA];
    faiss::idx_t *best = new faiss::idx_t[n_q * GAMMA];
    float total_score = 0.0f;

    for(int i = 0; i < n_q; ++ i) {
        int belong = labels[i];
        int ntotal = candidates[belong]->ntotal;
        int k = std::min(GAMMA, ntotal); 
        
        if (k > 0) {
            candidates[belong] -> search(1, _data_doc(q) + i * dim, k, scores + i * GAMMA, best + i * GAMMA);
            for (int j = 0; j < k; ++j) {
                total_score += scores[i * GAMMA + j];
            }
        }
    }
    delete[] scores;
    delete[] best;
    return total_score * INV_GAMMA;
}


static float CentroidMaxSim(MultiVectorDoc *q, MultiVectorDoc *d) {
    centroidIndex *qIndex = q -> index._c_index;
    faiss::Index *dIndex = d -> index._knn_index;
    if(qIndex == nullptr) qIndex = q -> index._c_index = centroidIndexing(q);
    if(dIndex == nullptr) dIndex = d -> index._knn_index = knnIndexing(d);
    faiss::Index **candidates;
    generate_candidates(qIndex -> n_centroid, qIndex -> centroids.get(), d, candidates);
    float score = reranking(q, candidates);
    for(int i = 0; i < qIndex -> n_centroid; ++ i) delete candidates[i];
    delete[] candidates;
    return score;
}

static float ConstCentroidMaxSim(const MultiVectorDoc *q, const MultiVectorDoc *d) {
    centroidIndex *qIndex = q -> index._c_index;
    faiss::Index *dIndex = d -> index._knn_index;
    faiss::Index **candidates;
    generate_candidates(qIndex -> n_centroid, qIndex -> centroids.get(), d, candidates);
    float score = reranking(q, candidates);
    for(int i = 0; i < qIndex -> n_centroid; ++ i) delete candidates[i];
    delete[] candidates;
    return score;
}

inline void generate_candidates_parallel(int n_centroid, float *centroids, const MultiVectorDoc *doc, faiss::Index** &candidates) {
    // std::cerr << "generating candidates" << std::endl;
    const int beta = std::min((int)doc -> doclen, std::max(GAMMA, (int)sqrt(doc -> doclen)));

    float *dis = new float[n_centroid * beta];
    faiss::idx_t *idx = new faiss::idx_t[n_centroid * beta];
    faiss::Index* knnIndex = doc -> index._knn_index;
    knnIndex -> search(n_centroid, centroids, beta, dis, idx);
    delete[] dis;

    candidates = new faiss::Index*[n_centroid];
    
    #pragma omp parallel for schedule(static) num_threads(std::min(n_centroid, omp_get_max_threads()))
    for(int i = 0; i < n_centroid; ++ i) {
        faiss::IndexFlatIP *idxc = new faiss::IndexFlatIP(doc -> dim);
        for(int j = 0; j < beta; ++ j) {
            int candidate_index = idx[i * beta + j];
            if(candidate_index == -1) break;
            idxc -> add(1, _data_doc(doc) + candidate_index * doc -> dim);
        }
        candidates[i] = idxc;
    }
    // std::cerr << "done" << std::endl;
    delete[] idx;
}

inline float reranking_parallel(const MultiVectorDoc* q, faiss::Index** candidates) {
    const int n_q = (int)q->doclen;
    const int dim = (int)q->dim;
    const int n_centroid = (int)q->index._c_index->n_centroid;
    const faiss::idx_t* labels = q->index._c_index->labels.get();
    const float* qdata = _data_doc(q);

    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(std::min(n_centroid, omp_get_max_threads()))
    for (int i = 0; i < n_q; ++i) {
        const int c = (int)labels[i];
        const float* qi = qdata + (size_t)i * dim;
        
        int ntotal = candidates[c]->ntotal;
        int k = std::min(GAMMA, ntotal);
        if (k > 0) {
            std::vector<float> s(k);
            std::vector<faiss::idx_t> a(k);
            candidates[c]->search(1, qi, k, s.data(), a.data());
            for (int j = 0; j < k; ++j) {
                sum += s[j];
            }
        }
    }
    return (float)(sum * INV_GAMMA);
}

static float ConstCentroidMaxSim_parallel(const MultiVectorDoc *q, const MultiVectorDoc *d) {
    centroidIndex *qIndex = q -> index._c_index;
    faiss::Index *dIndex = d -> index._knn_index;
    faiss::Index **candidates;
    generate_candidates_parallel(qIndex -> n_centroid, qIndex -> centroids.get(), d, candidates);
    // std::cerr << "reranking..." << std::endl;
    float score = reranking_parallel(q, candidates);
    // std::cerr << "done." << std::endl;
    for(int i = 0; i < qIndex -> n_centroid; ++ i) delete candidates[i];
    delete[] candidates;
    // std::cerr << "deleted" << ' ' << score << std::endl;
    return score;
}

static float ConstCentroidMaxSim_parallel_fused(const MultiVectorDoc *q, const MultiVectorDoc *d) {
    const centroidIndex *qIndex = q -> index._c_index;
    const faiss::Index *dIndex = d -> index._knn_index;
    const int n_q = (int)q->doclen;
    const int dim = (int)q->dim;
    const int n_centroid = (int)q->index._c_index->n_centroid;
    
    const int beta = std::min((int)d -> doclen, std::max(GAMMA, (int)sqrt(d -> doclen)));

    float *dis = new float[n_centroid * beta];
    faiss::idx_t *idx = new faiss::idx_t[n_centroid * beta];
    dIndex -> search(n_centroid, qIndex->centroids.get(), beta, dis, idx);
    delete[] dis;

    const faiss::idx_t* labels = q->index._c_index->labels.get();
    const float* qdata = _data_doc(q);
    const float *ddata = _data_doc(d);

    std::vector<std::vector<int>> qvec_id(n_centroid); qvec_id.reserve(n_centroid);
    for(int i = 0; i < n_q; ++ i) qvec_id[labels[i]].emplace_back(i);

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(std::min(n_centroid, omp_get_max_threads()))
    for(int i = 0; i < n_centroid; ++ i) {
        faiss::IndexFlatIP *idxc = new faiss::IndexFlatIP(d -> dim);
        for(int j = 0; j < beta; ++ j) {
            int candidate_index = idx[i * beta + j];
            if(candidate_index == -1) break;
            idxc -> add(1, ddata + candidate_index * d -> dim);
        }

        int ntotal = idxc->ntotal;
        int k = std::min(GAMMA, ntotal);
        if (k > 0) {
            std::vector<float> s(k);
            std::vector<faiss::idx_t> a(k);
            for (int qid : qvec_id[i]) {
                const float* qvec = qdata + qid * dim;
                idxc->search(1, qvec, k, s.data(), a.data());
                for (int j = 0; j < k; ++j) {
                    sum += s[j];
                }
            }
        }
        delete idxc;
    }

    delete[] idx;
    return (float)(sum * INV_GAMMA);
}

static float CentroidMaxSimWithHitItems(const MultiVectorDoc *q, const MultiVectorDoc *d, faiss::idx_t *item_buf, float *scores) {
    // 【重要提示】
    // 调用此函数前，传入的 item_buf 和 scores 数组大小必须按照 n_q * GAMMA 来分配
    const centroidIndex *qIndex = q -> index._c_index;
    const faiss::Index *dIndex = d -> index._knn_index;

    const int n_q = q -> doclen;
    const int n_d = d -> doclen;
    const int dim = q -> dim;
    const int n_centroid =  qIndex -> n_centroid;
    const float *qdata = _data_doc(q);
    const float *ddata = _data_doc(d);

    const int beta = std::min(n_d, std::max(GAMMA, (int)sqrt(n_d)));

    faiss::idx_t *idx = new faiss::idx_t[qIndex -> n_centroid * beta];
    float *dis = new float[qIndex -> n_centroid * beta];
    faiss::Index **candidates = new faiss::Index*[qIndex -> n_centroid];
    faiss::idx_t *vidmap = new faiss::idx_t[n_centroid * beta];

    dIndex -> search(qIndex -> n_centroid, qIndex -> centroids.get(), beta, dis, idx);
    delete[] dis;

    for(int i = 0; i < n_centroid; ++ i) {
        candidates[i] = new faiss::IndexFlatIP(d -> dim);
        for(int j = 0; j < beta; ++ j) {  
            vidmap[i*beta+j] = idx[i*beta+j];
            candidates[i] -> add(1, ddata + vidmap[i*beta+j] * d -> dim);
        }
    }
    delete[] idx;
    
    const faiss::idx_t *labels = q -> index._c_index -> labels.get();

    for(int i = 0; i < n_q; ++ i) {
        int belong = labels[i];
        int ntotal = candidates[belong]->ntotal;
        int k = std::min(GAMMA, ntotal);
        
        if (k > 0) {
            candidates[belong] -> search(1, qdata + i * dim, k, scores + i * GAMMA, item_buf + i * GAMMA);
            for (int j = 0; j < k; ++j) {
                item_buf[i * GAMMA + j] = vidmap[belong * beta + item_buf[i * GAMMA + j]];
            }
            for (int j = k; j < GAMMA; ++j) {
                scores[i * GAMMA + j] = 0.0f;
                item_buf[i * GAMMA + j] = -1;
            }
        } else {
            for (int j = 0; j < GAMMA; ++j) {
                scores[i * GAMMA + j] = 0.0f;
                item_buf[i * GAMMA + j] = -1;
            }
        }
    }
    float score = std::accumulate(scores, scores + n_q * GAMMA, 0.0f) * INV_GAMMA;

    for(int i = 0; i < qIndex -> n_centroid; ++ i) delete candidates[i];
    delete[] candidates;
    delete[] vidmap;
    
    return score;
}

// merge all vectors in a set of document into a single MultiVectorDoc object
MultiVectorDoc* merge_data(const std::vector<MultiVectorDoc*> &docs) {
    const int dim = (*docs.begin()) -> dim;
    int64_t n_vector = 0;
    for(MultiVectorDoc *doc : docs) n_vector += doc -> doclen;
    MultiVectorDoc *total = _make_doc(n_vector, dim);
    vdata_t *ptr = _data_doc(total);
    for(MultiVectorDoc *doc : docs) {
        int n_number = doc -> doclen * doc -> dim;
        memcpy(ptr, _data_doc(doc), n_number * sizeof(vdata_t));
        ptr += n_number;
    }
    return total;
}

#ifdef ENABLE_FIVECS
#define ENABLE_FIVECS
#include "FileIO.hpp"
#include "utils.hpp"

std::vector<MultiVectorDoc*> convertFromFivecs(std::vector<VectorDataType> &data_list, size_t n_doc) {
    std::vector<MultiVectorDoc*> vec;
    if(data_list.size() == 0) return vec;
    size_t dim = data_list[0].data.size();
    size_t n_vec = data_list.size(), l = 0;
    size_t i = 0;
    while(l < n_vec) {
        size_t r = l;
        for(; r + 1 < n_vec && data_list[r + 1].did == i; ++ r);
        MultiVectorDoc* ptr = _make_doc(r - l + 1, dim);
        vdata_t *arr = _data_doc(ptr);
        for(size_t j = 0; j <= r - l; ++ j)
            memcpy(arr + j * dim, data_list[j + l].data.data(), sizeof(vdata_t) * dim);
        vec.push_back(ptr);
        l = r + 1;
        ++ i;
    }
    return vec;
}

#endif

#endif