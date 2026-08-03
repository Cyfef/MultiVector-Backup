#include <iostream>
#include "FileIO.hpp"
#include "utils.hpp"
#include <vector>
#include <queue>
#include <set>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include <faiss/utils/distances.h>
#include <boost/program_options.hpp>
#include "space_multi.hpp"
#include "hnswlib/hnswlib/hnswlib.h"
#include "MultiVectorDoc.hpp"
#include "timecounter.hpp"

namespace bpo = boost::program_options;

size_t calcMaxLen(std::vector<MultiVectorDoc*> &docs) {
    size_t ans = 0;
    for(MultiVectorDoc* ptr : docs) ans = std::max(ans, ptr -> doclen);
    return ans;
}

int main(int argc, char **argv) {
    std::string setName;
    int M = 32;
    int ef_construction = 512;
    int dataScale;
    int dimension;

    namespace bpo = boost::program_options;
    try {
        bpo::options_description desc("Allowed options");
        desc.add_options()
        ("dataset", bpo::value<std::string>(&setName) -> required(), "Name of the dataset.");
        desc.add_options()
        ("M", bpo::value<int>(&M) -> required(), "HNSW arg M."); 
        desc.add_options()
        ("ef", bpo::value<int>(&ef_construction) -> required(), "HNSW arg ef_construction."); 
        desc.add_options()
        ("dataScale", bpo::value<int>(&dataScale) -> required(), "arg data scale of dataset."); 
        desc.add_options()
        ("dim", bpo::value<int>(&dimension) -> required(), "dimension of vectors."); 
        bpo::variables_map vmp;
        bpo::store(bpo::parse_command_line(argc, argv, desc), vmp);
        bpo::notify(vmp);
        if(vmp.count("help")) {
            std::cout << desc << std::endl;
        }
        if(vmp.count("dataset")) {
            std::cout << "SetName was set to " << setName << std::endl;
        } else {
            std::cout << "Error, no datasetName set." << std::endl;
        }
    } catch (const bpo::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    };

    const std::string dataPath = "/home/icypigeon/workspace/data/multivector/icde2025";
    const std::string docFileName = setName + "_" + std::to_string(dataScale) + "_" + std::to_string(dimension) + ".fivecs";
    const std::string documentFile(dataPath + "/document/" + setName + "/" + docFileName);
    const std::string queryFileName = setName + "_" + std::to_string(dimension) + "_query.fivecs";
    const std::string queryFile(dataPath + "/query/" + queryFileName);

    // const std::string logFile("/home/icypigeon/workspace/data/multivector/lotte/lotte_" + setName + "_forceHNSW_" + std::to_string(k) + "_output.txt");
    // const std::string timeFile("/home/icypigeon/workspace/data/multivector/lotte/lotte_" + setName + "_forceHNSW_" + std::to_string(k) + "_time.txt");

    std::vector<VectorDataType> dVec;
    std::vector<VectorDataType> qVec;

    // std::cout << "Reading Start..." << std::endl;

    auto[dim, numD] = ReadFromFivecs(documentFile, dVec);

    std::cout << "converting document vectors to MultiVectorDocs..." << std::endl;
    std::vector<MultiVectorDoc*> docs = convertFromFivecs(dVec, numD);

    dVec.clear();
    std::cout << "input done." << std::endl;

    size_t maxDocLen = calcMaxLen(docs);
    std::cout << "maxDocLen = " << maxDocLen << std::endl;

    

    int max_elements = numD;

    hnswlib::InnerProductSpace s_ip(dim);
    multihnsw::MultiVectorSpace s_multi(&s_ip, maxDocLen);
    hnswlib::HierarchicalNSW<vdata_t>* alg_hnsw = new hnswlib::HierarchicalNSW<vdata_t>(&s_multi, max_elements, M, ef_construction);

    // std::cout << numD << std::endl;
    // std::cout << s_multi.get_data_size() << std::endl;
    // std::cout << 279 * dim * sizeof(float) + sizeof(MultiVectorDoc) << std::endl;

    // std::cerr << docs[488] -> doclen << std::endl;
    // for(int i = 0; i < docs[488] -> doclen; ++ i) {
    //     for(int j =i *dim; j < (i + 1) * dim; ++ j) std::cerr << ((float*)_data_doc(docs[i]))[j] << ' ';
    //     std::cerr << std::endl;
    // }

    // alg_hnsw -> addPoint(docs[488], 488);
    // return 0;
    std::cout << "numD = " << numD << std::endl;
    

    TimeCounter timer;
    // auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Constructing docIndexes..." << std::endl;
    
    int t[4] = {};
    for(int i = 0; i < docs.size(); ++ i) {
        if(i == 9999) {
            t[1] = t[0];
        }
        if(i == 49999) {
            t[2] = t[0];
        }
        if(i == 99999) {
            t[3] = t[0];
        }
        timer.setTime();
        if(i % 1000 == 0) std::cout << i << "-th" << std::endl;
        docs[i] -> index._c_index = centroidIndexing(docs[i]);
        t[0] += timer.catchTime();
    }

    for(int i = 0; i < docs.size(); ++ i) {
        docs[i] -> index._knn_index = knnIndexing(docs[i]);
    }

    auto total_time = 0;
    
    for(int i = 0; i < docs.size(); ++ i) {
        if(i == 10000 || i == 50000 || i == 100000) {
            std::string hnsw_path = "/home/icypigeon/workspace/multivec/multiHNSW/script/advHNSW/advhnsw_" + setName + "_" + std::to_string(i) + "_" + std::to_string(dim) +  + "_" + std::to_string(M) + "_" + std::to_string(ef_construction) + "_g" + std::to_string(GAMMA) + ".bin";
            alg_hnsw -> saveIndex(hnsw_path);
            std::cout << "Index Saved at " << hnsw_path << std::endl;
            std::cout << "Construction time of " << i << " documents: \nHnsw part:" << total_time << "ms, Centroid indexing part: " << (i == 10000 ? t[1] : (i == 50000 ? t[2] : t[3])) << "ms." << std::endl;
        }
        if(i % 100 == 0) std::cout << "Inserting " << i << "th document vectors..." << std::endl;
        timer.setTime();
        alg_hnsw -> addPoint(docs[i], i);
        total_time += timer.catchTime();
    }
    std::cerr << "Construction done." << std::endl;
    std::string hnsw_path = "/home/icypigeon/workspace/multivec/multiHNSW/script/advHNSW/advhnsw_" + setName + "_" + std::to_string(dataScale) + "_" + std::to_string(dim) +  + "_" + std::to_string(M) + "_" + std::to_string(ef_construction) + "_g" + std::to_string(GAMMA) + ".bin";
    alg_hnsw -> saveIndex(hnsw_path);
    std::cout << "Index Saved at " << hnsw_path << std::endl;
    std::cout << "Construction time of " << numD << " documents: \nHnsw part:" << total_time << "ms, Centroid indexing part: " << t[0] << "ms." << std::endl;
    // std::cout << "Construction time: " << total_time << "ms." << std::endl;
}