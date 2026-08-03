// slice_fivecs_with_mvd.cpp
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

#include "/home/icypigeon/workspace/multivec/multivectorspace/utils.hpp"
#include "/home/icypigeon/workspace/multivec/multivectorspace/MultiVectorDoc.hpp"
#include "/home/icypigeon/workspace/multivec/multivectorspace/MDFileIO.hpp"  // 内含 fivecs_io::{LoadMultiVectorDocsFromFivecs_Streaming, SaveMultiVectorDocsToFivecs}

using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "用法: " << argv[0] << " <dataset> <dim>\n"
             << "示例: " << argv[0] << " gist 768\n";
        return 2;
    }
    const string dataset = argv[1];
    const string dim_arg = argv[2];

    const string prepath = "/home/icypigeon/workspace/data/multivector/icde2025/document/" + dataset + "/";
    const string in_path = prepath + dataset + "_200000_" + dim_arg + ".fivecs";
    try {
        // 1) 流式读取整份 fivecs → 一组 MultiVectorDoc*
        auto docs = fivecs_io::ReadMultiVectorDocFromFivecs(in_path);
        if (docs.empty()) {
            cerr << "输入为空或无法解析: " << in_path << "\n";
            return 1;
        }

        // 2) 校验文件内维度与命令行一致
        const size_t dim_file = docs.front()->dim;
        if (to_string(dim_file) != dim_arg) {
            cerr << "错误：文件内 dim=" << dim_file << " 与命令行参数 " << dim_arg << " 不一致。\n";
            return 1;
        }

        // 3) 逐个生成前缀 fivecs：{dataset}_{k}_{dim}.fivecs
        const size_t ndocs = docs.size();
        size_t generated = 0;
        for (size_t k = 10000; k <= 190000; k += 10000) {
            if (k > ndocs) break; // 源文档数不足，提前结束
            vector<MultiVectorDoc*> prefix;
            prefix.reserve(k);
            prefix.insert(prefix.end(), docs.begin(), docs.begin() + static_cast<long>(k));

            const string out_path = prepath + dataset + "_" + to_string(k) + "_" + dim_arg + ".fivecs";
            fivecs_io::SaveMultiVectorDocsToFivecs(out_path, prefix);
            cout << "[OK] 生成 " << out_path
                 << " （ndocs=" << k
                 << ", dim="   << dim_file
                 << "，nvecs自动统计）\n";
            ++generated;
        }

        if (generated == 0) {
            cerr << "源文件仅含 ndocs=" << ndocs << "，不足以切出前 10000 的前缀文件。\n";
            return 0;
        }
        return 0;
    } catch (const std::exception& e) {
        cerr << "异常：" << e.what() << "\n";
        return 3;
    }
}