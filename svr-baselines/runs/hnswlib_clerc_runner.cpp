#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "../hnswlib-master/hnswlib-master/hnswlib/hnswlib.h"

namespace {

using Clock = std::chrono::steady_clock;

struct FvecsData {
    std::vector<float> data;
    std::size_t count = 0;
    std::size_t dim = 0;
};

FvecsData read_fvecs(const std::string &path, std::size_t limit = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open fvecs file: " + path);
    }

    int32_t dim_i = 0;
    in.read(reinterpret_cast<char *>(&dim_i), sizeof(dim_i));
    if (!in) {
        throw std::runtime_error("failed to read first fvecs header: " + path);
    }
    if (dim_i <= 0) {
        throw std::runtime_error("invalid fvecs dim in: " + path);
    }
    const std::size_t dim = static_cast<std::size_t>(dim_i);

    in.seekg(0, std::ios::end);
    const std::size_t size = static_cast<std::size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    const std::size_t rec_size = sizeof(int32_t) + dim * sizeof(float);
    if (size % rec_size != 0) {
        throw std::runtime_error("invalid fvecs size: " + path);
    }
    std::size_t total = size / rec_size;
    if (limit != 0 && limit < total) {
        total = limit;
    }

    FvecsData out;
    out.count = total;
    out.dim = dim;
    out.data.resize(total * dim);

    for (std::size_t i = 0; i < total; ++i) {
        int32_t rec_dim = 0;
        in.read(reinterpret_cast<char *>(&rec_dim), sizeof(rec_dim));
        if (!in || rec_dim != dim_i) {
            throw std::runtime_error("inconsistent fvecs record in: " + path);
        }
        in.read(reinterpret_cast<char *>(out.data.data() + i * dim), dim * sizeof(float));
        if (!in) {
            throw std::runtime_error("short read in fvecs file: " + path);
        }
    }

    return out;
}

void write_results_tsv(
    const std::string &path,
    const std::vector<std::vector<std::pair<hnswlib::labeltype, float>>> &results) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open results path: " + path);
    }
    out.setf(std::ios::fixed);
    out.precision(10);
    for (std::size_t qi = 0; qi < results.size(); ++qi) {
        for (std::size_t r = 0; r < results[qi].size(); ++r) {
            out << qi << '\t' << results[qi][r].first << '\t' << (r + 1) << '\t' << (-results[qi][r].second) << '\n';
        }
    }
}

bool file_exists(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 12) {
        std::cerr
            << "usage: hnswlib_clerc_runner <base.fvecs> <query.fvecs> <index.bin> "
            << "<results.tsv> <num_queries> <k> <M> <ef_construction> <ef_search> "
            << "<build_threads> <search_threads>\n";
        return 1;
    }

    const std::string base_path = argv[1];
    const std::string query_path = argv[2];
    const std::string index_path = argv[3];
    const std::string results_path = argv[4];
    const std::size_t num_queries = static_cast<std::size_t>(std::stoull(argv[5]));
    const std::size_t k = static_cast<std::size_t>(std::stoull(argv[6]));
    const int M = std::stoi(argv[7]);
    const int ef_construction = std::stoi(argv[8]);
    const int ef_search = std::stoi(argv[9]);
    const int build_threads = std::stoi(argv[10]);
    const int search_threads = std::stoi(argv[11]);

    auto base = read_fvecs(base_path);
    auto queries = read_fvecs(query_path, num_queries);
    if (base.dim != queries.dim) {
        throw std::runtime_error("base/query dimension mismatch");
    }

    std::cerr << "base_count=" << base.count << " dim=" << base.dim << "\n";
    std::cerr << "query_count=" << queries.count << "\n";

    hnswlib::L2Space space(static_cast<int>(base.dim));
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;

    if (file_exists(index_path)) {
        std::cerr << "loading index from " << index_path << "\n";
        index.reset(new hnswlib::HierarchicalNSW<float>(&space, index_path, false, base.count));
    } else {
        std::cerr << "building index at " << index_path << "\n";
        index.reset(new hnswlib::HierarchicalNSW<float>(&space, base.count, M, ef_construction));
        omp_set_dynamic(0);
#pragma omp parallel for schedule(static) num_threads(build_threads)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(base.count); ++i) {
            index->addPoint(base.data.data() + static_cast<std::size_t>(i) * base.dim,
                            static_cast<hnswlib::labeltype>(i));
        }
        index->saveIndex(index_path);
    }

    index->setEf(ef_search);
    omp_set_dynamic(0);
    omp_set_num_threads(search_threads);

    std::vector<std::vector<std::pair<hnswlib::labeltype, float>>> results(queries.count);
    auto search_begin = Clock::now();
    for (std::size_t qi = 0; qi < queries.count; ++qi) {
        auto pq = index->searchKnn(queries.data.data() + qi * queries.dim, k);
        auto &query_results = results[qi];
        query_results.resize(pq.size());
        std::size_t out = query_results.size();
        while (!pq.empty()) {
            const auto item = pq.top();
            pq.pop();
            --out;
            query_results[out] = {item.second, item.first};
        }
    }
    auto search_end = Clock::now();
    const double search_sec = std::chrono::duration<double>(search_end - search_begin).count();

    write_results_tsv(results_path, results);

    std::cerr << "[Search Time] " << search_sec << "(sec)\n";
    std::cerr << "[QPS] " << (queries.count / search_sec) << "\n";
    return 0;
}
