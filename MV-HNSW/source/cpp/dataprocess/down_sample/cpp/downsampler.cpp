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

// Boost Program Options 用于解析命令行参数
#include <boost/program_options.hpp>

namespace fs = std::filesystem;
namespace bpo = boost::program_options;

/**
 * @brief 从 [0, n-1] 中随机抽取 m 个不重复的整数。
 * @param n 总体大小
 * @param m 样本大小
 * @return 包含 m 个随机整数的向量
 */
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
    
    std::sort(result.begin(), result.end()); // 排序便于后续处理
    return result;
}

/**
 * @brief 对大型 .fivecs 文件进行下采样
 * @param original_file_path 原始文件路径
 * @param output_file_path 输出文件路径
 * @param target_doc_count 目标文档数量
 */
void downsample_dataset(const std::string& original_file_path, const std::string& output_file_path, DidType target_doc_count) {
    std::clog << "\n=========================================================\n";
    std::clog << "Processing: " << original_file_path << "\n";
    std::clog << "Target documents: " << target_doc_count << "\n";
    std::clog << "Output to: " << output_file_path << "\n";
    std::clog << "=========================================================\n";

    // --- 准备阶段 ---
    std::clog << "[INFO] Reading header from original file..." << std::endl;
    auto [total_vecs, total_docs, dim] = ReadHeader(original_file_path);
    std::clog << "[INFO] Original file stats: " << total_docs << " docs, " << total_vecs << " vecs, " << dim << " dims." << std::endl;

    if (target_doc_count > total_docs) {
        std::cerr << "[WARNING] Target document count (" << target_doc_count 
                  << ") is greater than total documents (" << total_docs 
                  << "). Skipping this sample." << std::endl;
        return;
    }

    // --- 阶段一：元数据收集与计数 ---
    std::clog << "\n--- PASS 1: Metadata Collection & Counting ---\n";
    
    std::clog << "[INFO] Sampling " << target_doc_count << " document IDs..." << std::endl;
    std::vector<DidType> sampled_dids_vec = random_sample_dids(total_docs, target_doc_count);
    
    std::unordered_set<DidType> sampled_dids_set(sampled_dids_vec.begin(), sampled_dids_vec.end());
    std::unordered_map<DidType, DidType> old_did_to_new_did;
    for (DidType i = 0; i < target_doc_count; ++i) {
        old_did_to_new_did[sampled_dids_vec[i]] = i;
    }
    std::clog << "[SUCCESS] Sampled " << sampled_dids_set.size() << " unique document IDs." << std::endl;

    std::clog << "[INFO] Scanning file to count vectors for the new header..." << std::endl;
    VidType new_nvecs = 0;
    std::ifstream original_file_pass1(original_file_path, std::ios::binary);
    original_file_pass1.seekg(3 * sizeof(VidType)); // 跳过文件头

    for (VidType i = 0; i < total_vecs; ++i) {
        VidType vid;
        DidType did;
        original_file_pass1.read(reinterpret_cast<char*>(&vid), sizeof(VidType));
        original_file_pass1.read(reinterpret_cast<char*>(&did), sizeof(DidType));

        if (sampled_dids_set.count(did)) {
            new_nvecs++;
        }
        // 跳过向量数据以加速
        original_file_pass1.seekg(dim * sizeof(VectorDimensionType), std::ios::cur);
    }
    original_file_pass1.close();
    std::clog << "[SUCCESS] Pass 1 complete. The new dataset will have " << new_nvecs << " vectors." << std::endl;

    // --- 阶段二：数据过滤、重编号与写入 ---
    std::clog << "\n--- PASS 2: Filtering, Renumbering & Writing ---\n";
    
    std::ofstream output_file(output_file_path, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "[ERROR] Failed to open output file for writing: " << output_file_path << std::endl;
        return;
    }

    std::clog << "[INFO] Writing new header: " << new_nvecs << " vecs, " << target_doc_count << " docs, " << dim << " dims." << std::endl;
    output_file.write(reinterpret_cast<const char*>(&new_nvecs), sizeof(VidType));
    output_file.write(reinterpret_cast<const char*>(&target_doc_count), sizeof(DidType));
    output_file.write(reinterpret_cast<const char*>(&dim), sizeof(VidType));

    std::clog << "[INFO] Streaming, filtering, and writing vector data..." << std::endl;
    std::ifstream original_file_pass2(original_file_path, std::ios::binary);
    original_file_pass2.seekg(3 * sizeof(VidType)); // 再次跳过文件头

    std::vector<VectorDimensionType> vec_buffer(dim);
    VidType new_vid_count = 0;

    for (VidType i = 0; i < total_vecs; ++i) {
        VidType vid;
        DidType did;
        original_file_pass2.read(reinterpret_cast<char*>(&vid), sizeof(VidType));
        original_file_pass2.read(reinterpret_cast<char*>(&did), sizeof(DidType));
        original_file_pass2.read(reinterpret_cast<char*>(vec_buffer.data()), dim * sizeof(VectorDimensionType));

        if (sampled_dids_set.count(did)) {
            DidType new_did = old_did_to_new_did[did];
            
            output_file.write(reinterpret_cast<const char*>(&new_vid_count), sizeof(VidType));
            output_file.write(reinterpret_cast<const char*>(&new_did), sizeof(DidType));
            output_file.write(reinterpret_cast<const char*>(vec_buffer.data()), dim * sizeof(VectorDimensionType));
            
            new_vid_count++;
        }
        if ((i + 1) % 10000000 == 0) {
             std::clog << "[INFO] ... processed " << (i + 1) << " / " << total_vecs << " original vectors." << std::endl;
        }
    }
    original_file_pass2.close();
    output_file.close();

    std::clog << "[SUCCESS] Pass 2 complete. Wrote " << new_vid_count << " vectors to " << output_file_path << std::endl;
}

int main(int argc, char **argv) {
    std::string setName;

    try {
        bpo::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Produce help message")
            ("dataset", bpo::value<std::string>(&setName)->required(), "Name of the dataset (e.g., 'lifestyle').");
        
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

    // --- 修改开始 ---

    // 1. 定义顶层输出目录的路径
    // 由于可执行文件在 build/ 目录运行，".." 代表返回到项目根目录
    const fs::path top_level_output_dir = "../downsample_output";

    // 2. 根据数据集名称创建特定的子目录
    const fs::path dataset_output_dir = "/home/icypigeon/workspace/data/multivector/icde2025/document/" + setName +"/";

    try {
        // 创建主输出目录和数据集子目录（如果它们不存在）
        fs::create_directories(dataset_output_dir);
        std::clog << "[INFO] Output will be saved in: " << fs::absolute(dataset_output_dir) << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[ERROR] Could not create output directory: " << e.what() << std::endl;
        return 1;
    }

    // 3.1 构造原始文件的完整路径
    const std::string base_data_path = "/home/icypigeon/workspace/data/multivector/icde2025/document/" + setName +"/";
    const std::string original_file = base_data_path + setName + "_100000_768.fivecs";
    
    // 3.2 若对ms-macro下采样，采用下面的路径
    // const std::string base_data_path = "/home/icypigeon/workspace/data/multivector/ms-macro/";
    // const std::string original_file = base_data_path + "msmacro-" + setName + "-data.fivecs";

    // --- 修改结束 ---

    if (!fs::exists(original_file)) {
        std::cerr << "[ERROR] Original data file not found: " << original_file << std::endl;
        return 1;
    }

    const std::vector<DidType> target_doc_counts = {10000, 50000};

    for (const auto& count : target_doc_counts) {
        // --- 修改开始 ---
        // 4. 在新的目录结构下构造输出文件名
        const std::string output_filename = setName + "_" + std::to_string(count) + "_768.fivecs";
        const std::string output_file_path = (dataset_output_dir / output_filename).string();
        // --- 修改结束 ---
        
        try {
            downsample_dataset(original_file, output_file_path, count);
        } catch (const std::exception& e) {
            std::cerr << "[FATAL ERROR] An exception occurred while processing for target size " << count << ": " << e.what() << std::endl;
        }
    }
    
    std::clog << "\nAll downsampling tasks are complete." << std::endl;

    return 0;
}