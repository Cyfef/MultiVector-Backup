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
#include <faiss/IndexHNSW.h>
#include <boost/program_options.hpp>
// #include "space_multi_query.hpp"
#include "MultiVectorDoc.hpp"
// #include "hnswlib_query/hnswlib/hnswlib.h"
#include "timecounter.hpp"
#include "MDFileIO.hpp"
#include "neighbor_graph.hpp"

std::vector<int> timer;
TimeCounter timec;

namespace bpo = boost::program_options;

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
        ("dataScale", bpo::value<int>(&dataScale) -> required(), "arg data scale of dataset."); 
        desc.add_options()
        ("dim", bpo::value<int>(&dimension) -> required(), "dimension of vectors."); 
        desc.add_options()
        ("M", bpo::value<int>(&M) -> required(), "dimension of vectors."); 

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
    const std::string index_path("/home/icypigeon/workspace/multivec/multihnsw_v2/index/neighbor_graph");
    const std::string neighbor_graph_path(index_path + "/" + setName + "_" + std::to_string(dataScale) + "_" + std::to_string(dimension) + ".bin");

    std::cout << "Loading raw file from disk..." << std::endl;
    std::vector<MultiVectorDoc*> docs = fivecs_io::ReadMultiVectorDocFromFivecs(documentFile);
    std::cout << "done." << std::endl;

    std::vector<std::vector<std::pair<float, int>>> neighbor_graph;
    timec.setTime();
    neighbor_graph::build_neighbor_graph(docs, neighbor_graph, M = 32);
    int indexing_time = timec.catchTime();
    std::cout << "Indexing time: " << indexing_time << "ms." << std::endl;
    neighbor_graph::save_neighbor_graph(neighbor_graph_path, neighbor_graph);
    return 0;
}