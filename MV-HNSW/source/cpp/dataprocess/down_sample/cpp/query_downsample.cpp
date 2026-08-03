#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <filesystem>

#include "FileIO.hpp"
#include "utils.hpp"
#include <boost/program_options.hpp>

namespace fs = std::filesystem;
namespace bpo = boost::program_options;

// 从 [0, n-1] 中随机抽取 m 个不重复的整数。
std::vector<DidType> random_sample_dids(DidType n, DidType m) {
    if (m < 0 || m > n) {
        throw std::invalid_argument("Sample size m must be in [0, n]");
    }
    std::vector<DidType> universe(n);
    std::iota(universe.begin(), universe.end(), 0);

    std::vector<DidType> result;
    result.reserve(m);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::sample(universe.begin(), universe.end(),
                std::back_inserter(result),
                m,
                gen);
    
    std::sort(result.begin(), result.end());
    return result;
}


int main(int argc, char **argv) {
    std::string datasetName;
    int dim;
    int newQ_count;

    try {
        bpo::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Produce help message")
            ("dataset", bpo::value<std::string>(&datasetName)->required(), "Name of the dataset (e.g., 'lifestyle').")
            ("dim", bpo::value<int>(&dim)->required(), "Dimension of the vectors (e.g., 128).")
            ("newQ", bpo::value<int>(&newQ_count)->required(), "Number of queries to sample.");
        
        bpo::variables_map vmp;
        bpo::store(bpo::parse_command_line(argc, argv, desc), vmp);

        if (vmp.count("help")) {
            std::cout << desc << std::endl;
            return 1;
        }
        bpo::notify(vmp);
    } catch (const bpo::error& e) {
        std::cerr << "Error parsing command line: " << e.what() << std::endl;
        return 1;
    }

    // --- 路径构造 ---
    const std::string query_base_dir = "/home/icypigeon/workspace/data/multivector/icde2025/query/";
    const std::string source_query_filename = datasetName + "_" + std::to_string(dim) + "_query.fivecs";
    const std::string source_query_path = query_base_dir + source_query_filename;
    
    const std::string output_query_filename = datasetName + "_" + std::to_string(dim) + "_query_" + std::to_string(newQ_count) + ".fivecs";
    const std::string output_query_path = query_base_dir + output_query_filename;
    
    if (!fs::exists(source_query_path)) {
        std::cerr << "[ERROR] Source query file not found at: " << source_query_path << std::endl;
        return 1;
    }
    
    std::clog << "[INFO] Source Query File: " << source_query_path << std::endl;
    std::clog << "[INFO] Output Query File: " << output_query_path << std::endl;

    // --- 阶段一：元数据收集与计数 ---
    std::clog << "\n--- PASS 1: Metadata Collection & Counting ---\n";
    auto [total_vecs_source, total_queries_source, dim_source] = ReadHeader(source_query_path);
    assert(dim == dim_source);

    if (newQ_count > total_queries_source) {
        std::cerr << "[ERROR] Sample size (" << newQ_count << ") cannot be larger than total queries (" << total_queries_source << ")." << std::endl;
        return 1;
    }

    std::vector<DidType> sampled_qids_vec = random_sample_dids(total_queries_source, newQ_count);
    std::unordered_set<DidType> sampled_qids_set(sampled_qids_vec.begin(), sampled_qids_vec.end());
    std::unordered_map<DidType, DidType> old_qid_to_new_qid;
    for (DidType i = 0; i < newQ_count; ++i) {
        old_qid_to_new_qid[sampled_qids_vec[i]] = i;
    }

    VidType new_nvecs = 0;
    std::ifstream source_file_pass1(source_query_path, std::ios::binary);
    source_file_pass1.seekg(3 * sizeof(VidType)); // Skip header

    for (VidType i = 0; i < total_vecs_source; ++i) {
        VidType vid;
        DidType did; // Here, did represents query id
        source_file_pass1.read(reinterpret_cast<char*>(&vid), sizeof(VidType));
        source_file_pass1.read(reinterpret_cast<char*>(&did), sizeof(DidType));

        if (sampled_qids_set.count(did)) {
            new_nvecs++;
        }
        source_file_pass1.seekg(dim * sizeof(VectorDimensionType), std::ios::cur);
    }
    source_file_pass1.close();
    std::clog << "[SUCCESS] Pass 1 complete. The new query set will have " << new_nvecs << " vectors for " << newQ_count << " queries." << std::endl;

    // --- 阶段二：数据过滤、重编号与写入 ---
    std::clog << "\n--- PASS 2: Filtering, Renumbering & Writing ---\n";
    
    std::ofstream output_file(output_query_path, std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(&new_nvecs), sizeof(VidType));
    output_file.write(reinterpret_cast<const char*>(&newQ_count), sizeof(DidType));
    output_file.write(reinterpret_cast<const char*>(&dim), sizeof(VidType));

    std::ifstream source_file_pass2(source_query_path, std::ios::binary);
    source_file_pass2.seekg(3 * sizeof(VidType)); // Skip header

    std::vector<VectorDimensionType> vec_buffer(dim);
    VidType new_vid_count = 0;

    for (VidType i = 0; i < total_vecs_source; ++i) {
        VidType vid;
        DidType did;
        source_file_pass2.read(reinterpret_cast<char*>(&vid), sizeof(VidType));
        source_file_pass2.read(reinterpret_cast<char*>(&did), sizeof(DidType));
        source_file_pass2.read(reinterpret_cast<char*>(vec_buffer.data()), dim * sizeof(VectorDimensionType));

        if (sampled_qids_set.count(did)) {
            DidType new_did = old_qid_to_new_qid[did];
            
            output_file.write(reinterpret_cast<const char*>(&new_vid_count), sizeof(VidType));
            output_file.write(reinterpret_cast<const char*>(&new_did), sizeof(DidType));
            output_file.write(reinterpret_cast<const char*>(vec_buffer.data()), dim * sizeof(VectorDimensionType));
            
            new_vid_count++;
        }
    }
    source_file_pass2.close();
    output_file.close();

    std::clog << "[SUCCESS] Pass 2 complete. Wrote " << new_vid_count << " vectors to " << output_query_path << std::endl;
    
    return 0;
}