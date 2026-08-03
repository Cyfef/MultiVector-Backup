#ifndef FILE_IO_H
#define FILE_IO_H

#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <utility>
#include <unordered_set>
#include "utils.hpp"

std::tuple<VidType, DidType, unsigned> ReadHeader(const std::string &file_name) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int32_t nvecs, ndocs, dim;
    file.read(reinterpret_cast<char *>(&nvecs), sizeof(VidType));
    file.read(reinterpret_cast<char *>(&ndocs), sizeof(VidType));
    file.read(reinterpret_cast<char *>(&dim), sizeof(VidType));
    return std::make_tuple(nvecs, ndocs, dim);
}

std::pair<unsigned, DidType> ReadFromFivecs(const std::string &file_name, std::vector<VectorDataType> &data_list) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int32_t nvecs, ndocs, dim;
    file.read(reinterpret_cast<char *>(&nvecs), sizeof(VidType));
    file.read(reinterpret_cast<char *>(&ndocs), sizeof(VidType));
    file.read(reinterpret_cast<char *>(&dim), sizeof(VidType));
    std::cout << "#(vectors) = " << nvecs << ", #(docs) = " << ndocs << ", #(dim) = " << dim << std::endl;

    data_list.clear();
    // nvecs = std::min(nvecs, 10001);
    for (int i=0; i<nvecs; ++i) {
        VidType vid, did;
        file.read(reinterpret_cast<char*>(&vid), sizeof(VidType));
        file.read(reinterpret_cast<char*>(&did), sizeof(VidType));
        std::vector<VectorDimensionType> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim*sizeof(VectorDimensionType));
        if (!file) {
            throw std::runtime_error("Error reading file");
        }

        VectorDataType vector_data(vid, did, dim);
        for (int j = 0; j < dim; ++j)
            vector_data.data[j] = vec[j];
        if(vector_data.valid())
            data_list.emplace_back(vector_data);

        if (i%1000000 == 0)
            std::cout << "Read " << i << " vectors" << std::endl;
    }
    file.close();
    // ndocs = data_list.back().did + 1;
    return std::make_pair(static_cast<unsigned>(dim), ndocs);
}

std::pair<unsigned, DidType> ReadFromFivecs_filtered(const std::string &file_name, std::vector<VectorDataType> &data_list, std::unordered_set<DidType> &filter) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading vector data: " << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int32_t nvecs, ndocs, dim;
    file.read(reinterpret_cast<char *>(&nvecs), sizeof(VidType));
    file.read(reinterpret_cast<char *>(&ndocs), sizeof(VidType));
    file.read(reinterpret_cast<char *>(&dim), sizeof(VidType));
    // std::cout << "#(vectors) = " << nvecs << ", #(docs) = " << ndocs << ", #(dim) = " << dim << std::endl;

    data_list.clear();
    // nvecs = std::min(nvecs, 10001);
    for (int i=0; i<nvecs; ++i) {
        VidType vid, did;
        file.read(reinterpret_cast<char*>(&vid), sizeof(VidType));
        file.read(reinterpret_cast<char*>(&did), sizeof(VidType));

        if(filter.find(did) == filter.end())    // filter with sampled dids
            continue;

        std::vector<VectorDimensionType> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim*sizeof(VectorDimensionType));
        if (!file) {
            throw std::runtime_error("Error reading file");
        }

        VectorDataType vector_data(vid, did, dim);
        for (int j = 0; j < dim; ++j)
            vector_data.data[j] = vec[j];
        if(vector_data.valid())
            data_list.emplace_back(vector_data);

        if (i%1000000 == 0)
            std::cout << "Read " << i << " vectors" << std::endl;
    }
    file.close();
    return std::make_pair(static_cast<unsigned>(dim), ndocs);
}

void SaveToFivecs(const std::string &file_name, const std::vector<VectorDataType> &data_list, DidType ndocs) {
    // 如果列表为空，无法确定 dim，抛出异常或自行定义 dim=0 的行为
    if (data_list.empty()) {
        throw std::runtime_error("data_list 为空，无法写入 Fivecs 格式");
    }

    // 所有向量的维度应相同，取第一项作为 dim
    VidType nvecs = static_cast<VidType>(data_list.size());
    VidType dim   = static_cast<VidType>(data_list[0].data.size());

    std::ofstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing vector data: "
            << file_name << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 写入头：nvecs, ndocs, dim
    file.write(reinterpret_cast<const char *>(&nvecs), sizeof(VidType));
    file.write(reinterpret_cast<const char *>(&ndocs), sizeof(DidType));
    file.write(reinterpret_cast<const char *>(&dim),   sizeof(VidType));
    std::cout << "#(vectors) = " << nvecs
    << ", #(docs) = "  << ndocs
    << ", #(dim) = "   << dim   << std::endl;

    // 写入每一个向量记录：vid, did, data[]
    for (VidType i = 0; i < nvecs; ++i) {
    const auto &vec = data_list[i];
    file.write(reinterpret_cast<const char *>(&vec.vid), sizeof(VidType));
    file.write(reinterpret_cast<const char *>(&vec.did), sizeof(DidType));
    file.write(reinterpret_cast<const char *>(vec.data.data()),
        dim * sizeof(VectorDimensionType));
    if (!file) {
        throw std::runtime_error("Error writing file");
    }

    if (i % 1000000 == 0) {
        std::cout << "Wrote " << i << " vectors" << std::endl;
    }
    }

    file.close();
    std::cout << "Finished writing " << nvecs << " vectors to "
    << file_name << std::endl;
}

#endif
