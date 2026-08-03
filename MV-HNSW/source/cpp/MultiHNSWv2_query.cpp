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
#include "space_multi_query.hpp"
#include "MultiVectorDoc.hpp"
#include "hnswlib_query/hnswlib/hnswlib.h"
#include "timecounter.hpp"
#include "MDFileIO.hpp"
#include "neighbor_graph.hpp"

std::vector<int> timer;
TimeCounter timec;

namespace bpo = boost::program_options;

void printLogs(const std::string &filePath, const std::vector<std::vector<DidType>> &topKIdList, const std::vector<std::vector<VectorDimensionType>> &topKDisList) {
    std::ofstream log(filePath);
    std::cout << "Saving logfile into " << filePath << std::endl;
    log.setf(std::ios::fixed);
    log.precision(6);
    DidType numQ = topKIdList.size();
    unsigned k = topKIdList.cbegin() -> size();
    for(DidType queid = 0; queid < numQ; ++ queid) {
        log << queid << ' ' << k << std::endl;
        for(unsigned i = 0; i < k; ++ i) log << topKIdList[queid][i] << ' ';
        log << std::endl;
        for(unsigned i = 0; i < k; ++ i) log << topKDisList[queid][i] << ' ';
        log << std::endl;
    }
    std::cout << "Logfile saved successfully." << std::endl;
}

size_t calcMaxLen(std::vector<MultiVectorDoc*> &docs) {
    size_t ans = 0;
    for(MultiVectorDoc* ptr : docs) ans = std::max(ans, ptr -> doclen);
    return ans;
}

void printTimeLogs(const std::string &filePath) {
    std::ofstream log(filePath);
    for(int i = 0; i < timer.size(); ++ i) {
        log << i << ' ' << 0 << ' ' << timer[i] << std::endl;
    }
    std::cout << "Time file saved at " << filePath << std::endl;
}

int main(int argc, char **argv) {
    std::string setName;
    unsigned k;
    int dataScale;
    int dimension;
    int M;
    int ef_construction;
    int ef_search;

    namespace bpo = boost::program_options;
    try {
        bpo::options_description desc("Allowed options");
        desc.add_options()
        ("dataset", bpo::value<std::string>(&setName) -> required(), "Name of the dataset."); 
        desc.add_options()
        ("k", bpo::value<unsigned>(&k) -> required(), "Number of result vector."); 
        desc.add_options()
        ("dataScale", bpo::value<int>(&dataScale) -> required(), "arg data scale of dataset."); 
        desc.add_options()
        ("dim", bpo::value<int>(&dimension) -> required(), "dimension of vectors."); 
        desc.add_options()
        ("M", bpo::value<int>(&M) -> required(), "dimension of vectors."); 
        desc.add_options()
        ("efC", bpo::value<int>(&ef_construction) -> required(), "ef_construction."); 
        desc.add_options()
        ("efS", bpo::value<int>(&ef_search) -> required(), "ef_search.");

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

    const std::string queryFile("/home/icypigeon/workspace/data/multivector/icde2025/query/" + setName + "_" + std::to_string(dimension) + "_query.fivecs");

    std::vector<MultiVectorDoc*>queries = fivecs_io::ReadMultiVectorDocFromFivecs(queryFile);
    int numQ = queries.size();
    int dim = queries[0] -> dim;
    size_t maxDocLen = 300;

    hnswlib::InnerProductSpace s_ip(dim);
    multihnsw_query::MultiVectorSpace s_multi(&s_ip, maxDocLen);
    std::string hnsw_path = "/home/icypigeon/workspace/multivec/multiHNSW/script/advHNSW/advhnsw_" + setName + "_" + std::to_string(dataScale) + "_" + std::to_string(dim) +  + "_" + std::to_string(M) + "_" + std::to_string(ef_construction) + ".bin";
    std::cerr << "Loading hnsw from disk..." << std::endl;
    auto alg_hnsw = new hnswlib::HierarchicalNSW<float>(&s_multi, hnsw_path);
    alg_hnsw -> ef_ = ef_search;
    std::cerr << "Loading done." << std::endl;

    std::cout << "hnsw_size = " << alg_hnsw -> cur_element_count  << std::endl;
    std::cout << k << std::endl;

    std::cout << "Reloading Index of faiss::IndexFlatIP for bruteforce query..." << std::endl;
    for(int i = 0; i < alg_hnsw -> cur_element_count; ++ i) {
        MultiVectorDoc *p = static_cast<MultiVectorDoc*>((void*)(alg_hnsw -> getDataByInternalId(i)));
        p -> index._knn_index = knnIndexing(p);
    }
    std::cout << "done..." << std::endl;

    std::cout << "Loading neighbor graph..." << std::endl;
    std::string neighbor_graph_path = "/home/icypigeon/workspace/multivec/multihnsw_v2/index/neighbor_graph/" + setName + "_" + std::to_string(dataScale) + "_" + std::to_string(dim) + ".bin";
    std::vector<std::vector<std::pair<float,int>>> nb_gf;
    neighbor_graph::load_neighbor_graph(neighbor_graph_path, nb_gf);
    std::cout << "Done." << std::endl;

    faiss::idx_t *item_buf = new faiss::idx_t[maxDocLen * alg_hnsw -> max_elements_ * GAMMA];
    float *scores = new float[maxDocLen * alg_hnsw -> max_elements_ * GAMMA];

    std::vector<std::vector<DidType>> topKIDLists;
    std::vector<std::vector<dist_t>> topKDisLists;
    for(int i = 0; i < numQ; ++ i) {
        std::cerr << "Querying " << i << "th query..." << std::endl;
        timec.setTime();
        queries[i] -> index._c_index = centroidIndexing(queries[i]);
        // std::cerr << "Enter Searching..." << std::endl;
        std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result = alg_hnsw->searchKnn(queries[i], k, nullptr, &nb_gf, item_buf, scores);
        timer.push_back(timec.catchTime());
        // std::cerr << "Query done." << std::endl;
        std::vector<DidType> vec;
        std::vector<dist_t> vdis;
        while(!result.empty()) {
            vec.push_back(result.top().second);
            vdis.push_back(result.top().first);
            result.pop();
        }
        std::reverse(vec.begin(), vec.end());
        std::reverse(vdis.begin(), vdis.end());
        topKIDLists.push_back(vec);
        topKDisLists.push_back(vdis);
    }

    std::string logFile("/home/icypigeon/workspace/data/multivector/icde2025/log/MVHNSW/" + setName + "_MVHNSW_ndim" + std::to_string(dimension) + "_" + std::to_string(k) + "_" + std::to_string(dataScale) + "_" + std::to_string(M) + "_" + std::to_string(ef_construction) + "_" + std::to_string(ef_search) + "_g" + std::to_string(GAMMA) + "_output.txt");
    std::string timeFile("/home/icypigeon/workspace/data/multivector/icde2025/log/MVHNSW/" + setName + "_MVHNSW_ndim" + std::to_string(dimension) +  "_" + std::to_string(k) + "_" + std::to_string(dataScale) + "_" + std::to_string(M) + "_" + std::to_string(ef_construction) + "_" + std::to_string(ef_search) + "_g" + std::to_string(GAMMA) + "_" + "time.txt");

    printLogs(logFile, topKIDLists, topKDisLists);
    printTimeLogs(timeFile);
    return 0;
}