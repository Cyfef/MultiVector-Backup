#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <filesystem>

// 假设 FileIO.hpp 和 utils.hpp 在同一目录或include路径下
#include "FileIO.hpp"
#include "utils.hpp"

// Boost Program Options 用于解析命令行参数
#include <boost/program_options.hpp>

namespace fs = std::filesystem;
namespace bpo = boost::program_options;

/**
 * @brief 从一个较大的 .fivecs 文件创建一个较小的、嵌套的子集文件
 * @param source_file_path 源文件路径 (例如 200k 规模的文件)
 * @param output_file_path 输出文件路径 (例如 100k 规模的文件)
 * @param target_doc_count 目标文档数量 (例如 100000)
 */
void create_subset_file(const std::string& source_file_path, const std::string& output_file_path, DidType target_doc_count) {
    std::clog << "\n=========================================================\n";
    std::clog << "Creating subset from: " << source_file_path << "\n";
    std::clog << "Target documents: " << target_doc_count << "\n";
    std::clog << "Output will overwrite: " << output_file_path << "\n";
    std::clog << "=========================================================\n";

    // --- 准备阶段 ---
    std::clog << "[INFO] Reading header from source file..." << std::endl;
    auto [total_vecs_source, total_docs_source, dim] = ReadHeader(source_file_path);
    std::clog << "[INFO] Source file stats: " << total_docs_source << " docs, " << total_vecs_source << " vecs, " << dim << " dims." << std::endl;

    if (target_doc_count >= total_docs_source) {
        std::cerr << "[WARNING] Target document count (" << target_doc_count
                  << ") is not smaller than source document count (" << total_docs_source
                  << "). Skipping this size." << std::endl;
        return;
    }

    // --- 阶段一：元数据收集与计数 ---
    std::clog << "\n--- PASS 1: Metadata Collection & Counting ---\n";
    std::clog << "[INFO] Scanning file to count vectors for the new subset header..." << std::endl;
    VidType new_nvecs = 0;
    std::ifstream source_file_pass1(source_file_path, std::ios::binary);
    source_file_pass1.seekg(3 * sizeof(VidType)); // 跳过文件头

    for (VidType i = 0; i < total_vecs_source; ++i) {
        VidType vid;
        DidType did;
        source_file_pass1.read(reinterpret_cast<char*>(&vid), sizeof(VidType));
        source_file_pass1.read(reinterpret_cast<char*>(&did), sizeof(DidType));

        // 核心逻辑：只保留文档ID小于目标数量的向量 (前缀截取)
        if (did < target_doc_count) {
            new_nvecs++;
        }
        
        // 跳过向量数据以加速
        source_file_pass1.seekg(dim * sizeof(VectorDimensionType), std::ios::cur);
    }
    source_file_pass1.close();
    std::clog << "[SUCCESS] Pass 1 complete. The new subset will have " << new_nvecs << " vectors." << std::endl;

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
    std::ifstream source_file_pass2(source_file_path, std::ios::binary);
    source_file_pass2.seekg(3 * sizeof(VidType)); // 再次跳过文件头

    std::vector<VectorDimensionType> vec_buffer(dim);
    VidType new_vid_count = 0;

    for (VidType i = 0; i < total_vecs_source; ++i) {
        VidType vid;
        DidType did;
        source_file_pass2.read(reinterpret_cast<char*>(&vid), sizeof(VidType));
        source_file_pass2.read(reinterpret_cast<char*>(&did), sizeof(DidType));
        source_file_pass2.read(reinterpret_cast<char*>(vec_buffer.data()), dim * sizeof(VectorDimensionType));

        if (did < target_doc_count) {
            output_file.write(reinterpret_cast<const char*>(&new_vid_count), sizeof(VidType));
            output_file.write(reinterpret_cast<const char*>(&did), sizeof(DidType));
            output_file.write(reinterpret_cast<const char*>(vec_buffer.data()), dim * sizeof(VectorDimensionType));
            
            new_vid_count++;
        }

        if ((i + 1) % 10000000 == 0) {
             std::clog << "[INFO] ... processed " << (i + 1) << " / " << total_vecs_source << " original vectors." << std::endl;
        }
    }
    source_file_pass2.close();
    output_file.close();

    std::clog << "[SUCCESS] Pass 2 complete. Wrote " << new_vid_count << " vectors to " << output_file_path << std::endl;
}

int main(int argc, char **argv) {
    std::string datasetName;

    try {
        bpo::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Produce help message")
            ("dataset", bpo::value<std::string>(&datasetName)->required(), "Name of the dataset (e.g., 'lifestyle').");
        
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
    
    // 构造下采样文件所在的目录路径
    // 假设可执行文件在 build/ 目录运行, ".." 代表返回到项目根目录
    fs::path base_dir = "../downsample_output";
    fs::path dataset_dir = base_dir / datasetName;

    // 构造 200k 规模的源文件路径
    const std::string source_200k_filename = datasetName + "_200000.fivecs";
    const std::string source_file_path = (dataset_dir / source_200k_filename).string();

    if (!fs::exists(source_file_path)) {
        std::cerr << "[ERROR] Source 200k file not found at: " << source_file_path << std::endl;
        std::cerr << "[ERROR] Please ensure the largest downsampled file exists before running this script." << std::endl;
        return 1;
    }

    const std::vector<DidType> smaller_target_counts = {100000, 50000, 10000};

    for (const auto& count : smaller_target_counts) {
        // 构造输出文件的路径，指向同一个目录，实现覆盖
        const std::string output_filename = datasetName + "_" + std::to_string(count) + ".fivecs";
        const std::string output_filepath = (dataset_dir / output_filename).string();
        
        try {
            create_subset_file(source_file_path, output_filepath, count);
        } catch (const std::exception& e) {
            std::cerr << "[FATAL ERROR] An exception occurred while processing for target size " << count << ": " << e.what() << std::endl;
        }
    }
    
    std::clog << "\nAll nested downsampling tasks are complete." << std::endl;

    return 0;
}