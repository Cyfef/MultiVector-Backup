#ifndef SPACE_MULTI_QUERY
#define SPACE_MULTI_QUERY

#include "hnswlib_query_calc/hnswlib/hnswlib.h"
#include "MultiVectorDoc.hpp"
#include <utility>

using dist_t = float;
const dist_t INF = 1e9;

// template<typename MTYPE>
// struct MaxSimParam {
//     hnswlib::DISTFUNC<MTYPE> fstvectordistfunc_;
// };

namespace multihnsw_query {
    static dist_t
    BruteMaxSim(const void* pVec1, const void* pVec2, const void *param) {
 
        const MultiVectorDoc* doc1 = static_cast<const MultiVectorDoc*>(pVec1);
        const MultiVectorDoc* doc2 = static_cast<const MultiVectorDoc*>(pVec2);

        size_t dim = doc1 -> dim;
        size_t len1 = doc1 -> doclen;
        size_t len2 = doc2 -> doclen;

        const vdata_t *vec1 = _data_doc(doc1);
        const vdata_t *vec2 = _data_doc(doc2);
        
        // auto para = static_cast<const MaxSimParam<dist_t>*>(param);
        // int dim = para -> dim;  
        // const MultiVectorData* pv1 = static_cast<const MultiVectorData*>(pVec1);
        // const MultiVectorData* pv2 = static_cast<const MultiVectorData*>(pVec2);

        // if(pv1 -> n_vec_ == 0 || pv2 -> n_vec_ == 0) return -INF;     // no vector in either set.

        // const size_t num1 = pv1 -> n_vec_;
        // const size_t num2 = pv2 -> n_vec_;
        // const vdata_t* arr1 = pv1 -> arr;
        // const vdata_t* arr2 = pv2 -> arr;

        const hnswlib::DISTFUNC<dist_t> disfunc = *static_cast<const hnswlib::DISTFUNC<dist_t>*>(param);
        // std::cerr << len1 << ' ' << len2 << ' ' << data_size_ << std::endl;
        dist_t ans = 0;
        for(int i = 0; i < len1; ++ i) {
            dist_t roundMax = INF;
            for(int j = 0; j < len2; ++ j) {
                dist_t dist_ = disfunc(vec1 + i * dim, vec2 + j * dim, &dim);
                roundMax = std::min(roundMax, dist_);   // disfunc -> 1 - dot --> min, in space_ip
            }
            ans += roundMax;
        }
        return ans;
    }

    static dist_t
    AveMaxSimDistance(const void* pVec1, const void *pVec2, const void *param) {
        // std::cerr << "query_ave" << std::endl;
        const MultiVectorDoc *q = static_cast<const MultiVectorDoc*>(pVec1);
        const MultiVectorDoc *d = static_cast<const MultiVectorDoc*>(pVec2);
        // assert(static_cast<const MultiVectorDoc*>(pVec1) -> doclen && static_cast<const MultiVectorDoc*>(pVec2) -> doclen);
        // std::cerr << "here1" << std::endl;
        dist_t ans = ConstCentroidMaxSim(q, d) / q -> doclen;
        ans += ConstCentroidMaxSim(d, q) / d -> doclen;
        // std::cerr << "here2" << std::endl;
        return (2 - ans) / 2;
    }

    static dist_t
    UniformedSingleMaxSim(const void *pVec1, const void *pVec2, const void *param) {
        const MultiVectorDoc *q = static_cast<const MultiVectorDoc*>(pVec1);
        const MultiVectorDoc *d = static_cast<const MultiVectorDoc*>(pVec2);
        float ret = 1 - ConstCentroidMaxSim(q, d) / q -> doclen;
        // std::cerr << "done." << std::endl;
        return ret;
    }


    class MultiVectorSpace : public hnswlib::AdvancedSpaceInterface<dist_t> {
        hnswlib::DISTFUNC<dist_t> fstdistfunc_;
        hnswlib::DISTFUNC<dist_t> fstdistfunc_2;
        hnswlib::DISTFUNC<dist_t> fstvectordistfunc_;
        // hnswlib::SpaceInterface<dist_t> *singleVectorSpace;
        size_t data_size_;
        // size_t vdata_len_;
        // MaxSimParam<dist_t> param;
    public:
        MultiVectorSpace(hnswlib::SpaceInterface<dist_t>* singleVectorSpace, size_t vdata_len) {
            fstdistfunc_ = AveMaxSimDistance;
            fstdistfunc_2 = UniformedSingleMaxSim;
            fstvectordistfunc_ = singleVectorSpace -> get_dist_func();
            size_t dim = *static_cast<size_t*>(singleVectorSpace -> get_dist_func_param());
            // vdata_len_ = vdata_len;
            data_size_ = _size_doc(vdata_len, dim);
        }

        size_t get_data_size() {
            return data_size_;
        }

        hnswlib::DISTFUNC<dist_t> get_dist_func() {
            return fstdistfunc_2;
        }

        void* get_dist_func_param() {
            return &fstvectordistfunc_;
        }
        hnswlib::DISTFUNC<dist_t> get_dist_func_construct() {
            return fstdistfunc_2;
        }
        hnswlib::DISTFUNC<dist_t> get_dist_func_search() {
            return fstdistfunc_2;
        }
    };
}

#endif