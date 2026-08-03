#ifndef COLBERT_UTILS_H
#define COLBERT_UTILS_H

#include <cmath>
#include <vector>

typedef int VidType;
typedef int DidType;
typedef float VectorDimensionType;
struct VectorDataType {
    VidType vid;  
    DidType did;
    std::vector<VectorDimensionType> data;

    VectorDataType(VidType _vid=0, DidType _did=0, size_t dim=0): vid(_vid), did(_did) {
        data.reserve(dim);
        data.resize(dim);
        std::fill(data.begin(), data.end(), (VectorDimensionType)0);
    }

    bool valid() {
        int dim = data.size();
        for(int i = 0; i < dim; ++ i)
            if(std::isnan(data[i]))
                return false;
        return true;
    }
};

struct CompressedVectorDataType {
    VidType vid;
    DidType did;
    VidType cid;
    std::vector<unsigned char> data;
};

using faissVectorDimensionType = float;
using ScoreType = float;



struct MetaData {
    DidType docid;
    bool enable;
    MetaData(DidType docid = -1, bool enable = false) : docid(docid), enable(enable) {
    }
};

const int DB_BUFFERSIZE = 1000000;

const int WARP_NITER = 10;

const int WARP_COMPRESS_B = 4;
const int CompressRate = 32 / WARP_COMPRESS_B;
#endif