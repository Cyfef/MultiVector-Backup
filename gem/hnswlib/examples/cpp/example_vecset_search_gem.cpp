#include <iostream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <immintrin.h>
#include <execinfo.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <omp.h>
#include <complex>
#include <cblas.h>
#include <chrono>  
#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/space_l2.h"
#include "../../hnswlib/vectorset.h"
#include "../../cnpy/cnpy.h"
// #include <experimental/filesystem>

extern "C" void openblas_set_num_threads(int num_threads);

int NUM_CLUSTER_CALC_RUNTIME = 262144;
const float* materialize_vectorset_data(const vectorset& vectors,
                                        std::vector<float>& decoded_buffer);
void ensure_directory_exists(const std::string& path);
void install_fatal_signal_handlers();

#define TEST_MSMARCO 0
#define TEST_LOTTE 1
#define TEST_OKVQA 2
#define TEST_EVQA 3
#define TEST_SCIDOCS 4
#define TEST_GENERIC 5

int dataset = TEST_MSMARCO; // 0: MSMARCO, 1: LOTTE, 2: OKVQA, 3: EVQA, 4: SCIDOCS, 5: generic ColBERT NPY
std::string dataset_path = "../../gem_data/";

int VECTOR_DIM = 128;

constexpr int MSMACRO_TEST_NUMBER = 354;
constexpr int LOTTE_TEST_NUMBER = 98;
constexpr int OKVQA_TEST_NUMBER = 5;
constexpr int EVQA_TEST_NUMBER = 3;
constexpr int SCIDOCS_TEST_NUMBER = 2;

constexpr int NUM_BASE_SETS_MS = 25000 * MSMACRO_TEST_NUMBER;
constexpr int NUM_QUERT_MS = 6980;
constexpr int NUM_CLUSTER_MS = 262144;
constexpr int NUM_GRAPH_CLUSTER_MS = 40960;

constexpr long long NUM_BASE_VECTOR_LOTTE = 339419977;
constexpr int NUM_BASE_SETS_LOTTE = 2428853;
constexpr int NUM_QUERT_LOTTE = 2930;
constexpr int NUM_CLUSTER_LOTTE = 262144;
constexpr int NUM_GRAPH_CLUSTER_LOTTE = 10240;

constexpr long long NUM_BASE_VECTOR_OKVQA = 14119353;
constexpr int NUM_BASE_SETS_OKVQA = 114809;
constexpr int NUM_QUERT_OKVQA = 5046;
constexpr int NUM_CLUSTER_OKVQA = 32768;
constexpr int NUM_GRAPH_CLUSTER_OKVQA = 1024;

constexpr long long NUM_BASE_VECTOR_EVQA = 9745953;
constexpr int NUM_BASE_SETS_EVQA = 51472;
constexpr int NUM_QUERT_EVQA = 3750;
constexpr int NUM_CLUSTER_EVQA = 32768;
constexpr int NUM_GRAPH_CLUSTER_EVQA = 1024;

constexpr long long NUM_BASE_VECTOR_SCIDOCS = 580970;
constexpr int NUM_BASE_SETS_SCIDOCS = 25657;
constexpr int NUM_QUERT_SCIDOCS = 1000;
constexpr int NUM_CLUSTER_SCIDOCS = 4096;
constexpr int NUM_GRAPH_CLUSTER_SCIDOCS = 512;
 
constexpr int CPU_num = 1;
constexpr int M_index = 24;
int EF_index = 32;


int NUM_BASE_SETS = 25000 * MSMACRO_TEST_NUMBER;
int NUM_QUERY_SETS = 6980;
int NUM_CLUSTER = 262144;
int NUM_GRAPH_CLUSTER = 40960;
int QUERY_VECTOR_COUNT = 32;

// 320
int NPROB = 4;
constexpr int K = 100;
int rerankK = 512;

// 5
// std::vector<int>eflist = {10, 32, 64, 100, 200, 400, 800, 1000, 2000, 4000, 6000, 10000, 15000, 20000, 40000};
// std::vector<int>eflist = {2000};
// std::vector<int>eflist = {2000, 4000, 8000, 10000, 15000, 20000, 30000};
// 10
// std::vector<int>eflist = {1000};
// std::vector<int>eflist = {10, 32, 64, 100, 200, 400, 800, 2000, 4000, 8000, 10000, 15000, 20000, 30000};
// std::vector<int>eflist = {100, 256, 512, 1000, 2000, 4000, 80000, 16000, 24000, 32000};
std::vector<int>eflist = {1000, 2000, 4000, 8000, 16000, 24000, 40000, 50000};
// 100
// std::vector<int>eflist = {10, 32, 100, 200, 400, 800, 1000, 2000, 4000, 8000, 16000, 24000, 40000, 50000};
// std::vector<int>eflist = {80000, 120000, 150000, 200000};
//std::vector<int>eflist = {4000};

struct QuerySearchProfile {
    int query_vecnum = 0;
    size_t selected_unique_cluster_count = 0;
    size_t entry_point_count = 0;
    size_t merged_cluster_membership_sum = 0;
    size_t merged_routed_doc_union_size = 0;
    size_t graph_candidate_count = 0;
    size_t rerank_candidate_count = 0;
    size_t final_result_count = 0;
    double routing_time_seconds = 0.0;
    double graph_search_time_seconds = 0.0;
    double rerank_time_seconds = 0.0;
    double total_time_seconds = 0.0;
    std::vector<int> selected_unique_cluster_ids;
    std::vector<size_t> selected_unique_cluster_sizes;
    std::vector<std::vector<int>> per_vector_clusters;
    std::vector<std::vector<size_t>> per_vector_cluster_sizes;
};

std::vector<size_t> read_npy_shape(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open NPY file: " + path);
    }

    char magic[6];
    file.read(magic, sizeof(magic));
    if (!file || std::string(magic, sizeof(magic)) != "\x93NUMPY") {
        throw std::runtime_error("Invalid NPY header: " + path);
    }

    unsigned char major = 0;
    unsigned char minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t short_header_len = 0;
        file.read(reinterpret_cast<char*>(&short_header_len), sizeof(short_header_len));
        header_len = short_header_len;
    } else if (major == 2 || major == 3) {
        file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    } else {
        throw std::runtime_error("Unsupported NPY version in: " + path);
    }

    std::string header(header_len, '\0');
    file.read(&header[0], header_len);

    const size_t shape_key_pos = header.find("'shape'");
    if (shape_key_pos == std::string::npos) {
        throw std::runtime_error("NPY shape missing in header: " + path);
    }

    const size_t open_paren_pos = header.find('(', shape_key_pos);
    const size_t close_paren_pos = header.find(')', open_paren_pos);
    if (open_paren_pos == std::string::npos || close_paren_pos == std::string::npos) {
        throw std::runtime_error("Unable to parse NPY shape: " + path);
    }

    std::stringstream shape_stream(header.substr(open_paren_pos + 1, close_paren_pos - open_paren_pos - 1));
    std::vector<size_t> shape;
    std::string token;
    while (std::getline(shape_stream, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(),
                                   [](unsigned char ch) { return std::isspace(ch) != 0; }),
                    token.end());
        if (!token.empty()) {
            shape.push_back(static_cast<size_t>(std::stoull(token)));
        }
    }

    if (shape.empty()) {
        throw std::runtime_error("Empty NPY shape in header: " + path);
    }
    return shape;
}

size_t get_npy_element_count(const std::string& path) {
    std::vector<size_t> shape = read_npy_shape(path);
    size_t count = 1;
    for (size_t dim : shape) {
        count *= dim;
    }
    return count;
}

int infer_vector_dim(const std::string& base_embedding_path, const std::string& query_embedding_path) {
    const std::vector<size_t> base_shape = read_npy_shape(base_embedding_path);
    const std::vector<size_t> query_shape = read_npy_shape(query_embedding_path);
    if (base_shape.size() < 2 || query_shape.size() < 2) {
        throw std::runtime_error("Embedding arrays must be at least 2D.");
    }

    const int base_dim = static_cast<int>(base_shape.back());
    const int query_dim = static_cast<int>(query_shape.back());
    if (base_dim != query_dim) {
        throw std::runtime_error("Base/query embedding dimension mismatch: " +
                                 std::to_string(base_dim) + " vs " + std::to_string(query_dim));
    }
    return base_dim;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path.c_str(), std::ios::binary);
    return file.good();
}

void ensure_directory_exists(const std::string& path) {
    if (path.empty()) {
        return;
    }

    std::string normalized = path;
    while (!normalized.empty() && normalized.back() == '/') {
        normalized.pop_back();
    }
    if (normalized.empty()) {
        return;
    }

    size_t pos = 0;
    if (normalized[0] == '/') {
        pos = 1;
    }

    while (pos <= normalized.size()) {
        pos = normalized.find('/', pos);
        const std::string current = normalized.substr(0, pos);
        if (!current.empty()) {
            struct stat st {};
            if (stat(current.c_str(), &st) != 0) {
                if (mkdir(current.c_str(), 0775) != 0 && errno != EEXIST) {
                    throw std::runtime_error("Failed to create directory: " + current + " errno=" + std::to_string(errno));
                }
            } else if (!S_ISDIR(st.st_mode)) {
                throw std::runtime_error("Path exists but is not a directory: " + current);
            }
        }
        if (pos == std::string::npos) {
            break;
        }
        ++pos;
    }
}

void fatal_signal_handler(int signal_number) {
    const char header[] = "\nFatal signal received in example_vecset_search_gem\n";
    (void)!write(STDERR_FILENO, header, sizeof(header) - 1);
    void* frames[64];
    const int frame_count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
    _exit(128 + signal_number);
}

void install_fatal_signal_handlers() {
    struct sigaction action {};
    sigemptyset(&action.sa_mask);
    action.sa_handler = fatal_signal_handler;
    action.sa_flags = SA_RESETHAND;
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGABRT, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
    sigaction(SIGFPE, &action, nullptr);
    sigaction(SIGILL, &action, nullptr);
}

std::pair<int, int> infer_query_layout(const std::string& query_embedding_path) {
    const std::vector<size_t> query_shape = read_npy_shape(query_embedding_path);
    if (query_shape.size() != 3) {
        throw std::runtime_error("Expected a 3D query embedding array: " + query_embedding_path);
    }
    return {static_cast<int>(query_shape[0]), static_cast<int>(query_shape[1])};
}

std::pair<int, int> infer_query_layout_from_lengths(const std::string& query_length_path) {
    cnpy::NpyArray qlens_npy = cnpy::npy_load(query_length_path);
    int q_num = static_cast<int>(qlens_npy.shape[0]);
    const int64_t* qlens_data = qlens_npy.data<int64_t>();
    int max_query_vectors = 0;
    for (int i = 0; i < q_num; ++i) {
        max_query_vectors = std::max(max_query_vectors, static_cast<int>(qlens_data[i]));
    }
    return {q_num, max_query_vectors};
}

long long infer_total_vectors_from_shards(const std::string& docdata_path, int shard_count) {
    long long total_vectors = 0;
    for (int i = 0; i < shard_count; ++i) {
        const std::string embfile_name = docdata_path + "encoding" + std::to_string(i) + "_float16.npy";
        if (!file_exists(embfile_name)) {
            continue;
        }
        const std::vector<size_t> shape = read_npy_shape(embfile_name);
        if (shape.size() < 2) {
            throw std::runtime_error("Expected 2D shard embedding array: " + embfile_name);
        }
        total_vectors += static_cast<long long>(shape[0]);
    }
    return total_vectors;
}

int infer_total_docs_from_shards(const std::string& docdata_path, int shard_count) {
    int total_docs = 0;
    for (int i = 0; i < shard_count; ++i) {
        const std::string lensfile_name = docdata_path + "doclens" + std::to_string(i) + ".npy";
        if (!file_exists(lensfile_name)) {
            continue;
        }
        const std::vector<size_t> shape = read_npy_shape(lensfile_name);
        if (shape.empty()) {
            throw std::runtime_error("Expected 1D doc length array: " + lensfile_name);
        }
        total_docs += static_cast<int>(shape[0]);
    }
    return total_docs;
}

int infer_doc_shard_count(const std::string& docdata_path) {
    int shard_count = 0;
    while (file_exists(docdata_path + "doclens" + std::to_string(shard_count) + ".npy")) {
        ++shard_count;
    }
    if (shard_count == 0) {
        throw std::runtime_error("No doclens*.npy shards found in " + docdata_path);
    }
    return shard_count;
}

std::string index_metadata_path(const std::string& index_dir) {
    return index_dir + "metadata.txt";
}

void write_index_metadata(
    const std::string& index_dir,
    int vector_dim,
    int num_base_sets,
    int num_query_sets,
    int num_cluster,
    int num_graph_cluster) {
    std::ofstream out(index_metadata_path(index_dir), std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write index metadata: " + index_metadata_path(index_dir));
    }
    out << "vector_dim " << vector_dim << '\n';
    out << "num_base_sets " << num_base_sets << '\n';
    out << "num_query_sets " << num_query_sets << '\n';
    out << "num_cluster " << num_cluster << '\n';
    out << "num_graph_cluster " << num_graph_cluster << '\n';
}

void validate_index_metadata_or_throw(
    const std::string& index_dir,
    int vector_dim,
    int num_base_sets,
    int num_query_sets,
    int num_cluster,
    int num_graph_cluster) {
    const std::string metadata_path = index_metadata_path(index_dir);
    if (!file_exists(metadata_path)) {
        throw std::runtime_error(
            "Index metadata not found at " + metadata_path +
            ". Refuse to load a potentially stale index; rerun with GEM_REBUILD=1.");
    }

    std::ifstream in(metadata_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open index metadata: " + metadata_path);
    }

    std::unordered_map<std::string, long long> values;
    std::string key;
    long long value = 0;
    while (in >> key >> value) {
        values[key] = value;
    }

    const auto require_match = [&](const std::string& name, long long expected) {
        auto it = values.find(name);
        if (it == values.end()) {
            throw std::runtime_error(
                "Index metadata is missing '" + name + "' in " + metadata_path +
                ". Refuse to load a potentially stale index; rerun with GEM_REBUILD=1.");
        }
        if (it->second != expected) {
            throw std::runtime_error(
                "Index metadata mismatch for " + name + ": saved=" + std::to_string(it->second) +
                " current=" + std::to_string(expected) +
                ". Refuse to load a stale index; rerun with GEM_REBUILD=1.");
        }
    };

    require_match("vector_dim", vector_dim);
    require_match("num_base_sets", num_base_sets);
    require_match("num_query_sets", num_query_sets);
    require_match("num_cluster", num_cluster);
    require_match("num_graph_cluster", num_graph_cluster);
}

int get_env_int_or_default(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    return std::stoi(value);
}

bool get_env_bool_or_default(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    const std::string parsed(value);
    if (parsed == "1" || parsed == "true" || parsed == "TRUE" || parsed == "yes" || parsed == "YES") {
        return true;
    }
    if (parsed == "0" || parsed == "false" || parsed == "FALSE" || parsed == "no" || parsed == "NO") {
        return false;
    }
    throw std::runtime_error("Invalid boolean environment value for " + std::string(name) + ": " + parsed);
}

std::string get_env_string_or_default(const char* name, const std::string& default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    return std::string(value);
}

std::vector<int> get_env_int_list_or_default(const char* name, const std::vector<int>& default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }

    std::string text(value);
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream stream(text);
    std::vector<int> parsed;
    int item = 0;
    while (stream >> item) {
        if (item <= 0) {
            throw std::runtime_error("Invalid non-positive integer in " + std::string(name) + ": " + std::to_string(item));
        }
        parsed.push_back(item);
    }
    if (parsed.empty()) {
        throw std::runtime_error("Invalid empty integer list in " + std::string(name) + ": " + std::string(value));
    }
    return parsed;
}

double get_peak_ram_mebibytes() {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return -1.0;
    }
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
}

double get_current_ram_mebibytes() {
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        return -1.0;
    }
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long long kib = 0;
            iss >> kib;
            if (iss.fail()) {
                return -1.0;
            }
            return static_cast<double>(kib) / 1024.0;
        }
    }
    return -1.0;
}

void load_qrels_if_exists(const std::string& qrelfile_name, size_t q_num, std::vector<std::vector<int>>& qrels) {
    qrels.resize(q_num + 1);
    if (!file_exists(qrelfile_name)) {
        std::cout << "Skip qrels load: file not found: " << qrelfile_name << std::endl;
        return;
    }

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int num1, num2;
        if (iss >> num1 >> num2) {
            if (num1 >= 0 && num1 < static_cast<int>(q_num)) {
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();
}

void get_unique_top_k_indices_col(const std::vector<float>& matrix, int rows, int cols, int topk, std::unordered_set<int>& unique_indices) {
    // std::cout << matrix.size() << std::endl;
    std::vector<int> all_scores(rows * topk);
    // #pragma omp parallel for
    // std::cout << matrix.size() << std::endl;
    // std::cout << rows << std::endl;
    // std::cout << cols << std::endl;
    // std::cout << topk << std::endl;
    // std::cout << all_scores.size() << std::endl;
    #pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        // std::cout << row << std::endl;
        std::vector<std::pair<float, int>> scores(cols);
        // 提取该列数据
        for (int col = 0; col < cols; ++col) {
            // std::cout << row << " " << col << std::endl;
            scores[col] = {matrix[row * cols + col], col};  
        }

        // 仅排序前 K 个最大元素
        std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first; // 降序
                          });

        // 直接存入集合去重
        for (int i = 0; i < topk; ++i) {
            // std::cout << row << " " << row * topk + i << " " << scores[i].first << " " << scores[i].second << std::endl;
            all_scores[row * topk + i] = scores[i].second;
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < topk; ++i) {
            // std::cout << row << " " << i << std::endl;
            unique_indices.insert(all_scores[row * topk + i]);
        }
    }
    // for(int t: unique_indices) {
    //     std::cout<< t << " ";
    // }
    // std::cout << std::endl;
    return;
}

void get_top_k_indices_col_with_details(const std::vector<float>& matrix,
                                        int rows,
                                        int cols,
                                        int topk,
                                        std::vector<std::vector<int>>& per_row_indices,
                                        std::unordered_set<int>& unique_indices) {
    per_row_indices.assign(rows, std::vector<int>(topk, -1));
    #pragma omp parallel for
    for (int row = 0; row < rows; ++row) {
        std::vector<std::pair<float, int>> scores(cols);
        for (int col = 0; col < cols; ++col) {
            scores[col] = {matrix[row * cols + col], col};
        }

        std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                              return a.first > b.first;
                          });

        for (int i = 0; i < topk; ++i) {
            per_row_indices[row][i] = scores[i].second;
        }
    }

    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < topk; ++i) {
            unique_indices.insert(per_row_indices[row][i]);
        }
    }
}

std::string join_int_vector(const std::vector<int>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << values[i];
    }
    return oss.str();
}

std::string join_size_vector(const std::vector<size_t>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << values[i];
    }
    return oss.str();
}

std::string join_nested_int_vectors(const std::vector<std::vector<int>>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ';';
        }
        oss << join_int_vector(values[i]);
    }
    return oss.str();
}

std::string join_nested_size_vectors(const std::vector<std::vector<size_t>>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ';';
        }
        oss << join_size_vector(values[i]);
    }
    return oss.str();
}

std::string make_query_profile_path(const std::string& run_result_file) {
    const size_t ext_pos = run_result_file.rfind(".tsv");
    if (ext_pos != std::string::npos) {
        return run_result_file.substr(0, ext_pos) + ".query_profile.tsv";
    }
    return run_result_file + ".query_profile.tsv";
}

void write_query_profile_header(std::ofstream& stream) {
    stream
        << "query_idx\tquery_vecnum\tnprob\tef\trerankK\tselected_unique_cluster_count\tentry_point_count\t"
        << "selected_unique_cluster_ids\tselected_unique_cluster_sizes\tper_vector_clusters\t"
        << "per_vector_cluster_sizes\tmerged_cluster_membership_sum\tmerged_routed_doc_union_size\t"
        << "graph_candidate_count\trerank_candidate_count\tfinal_result_count\trouting_time_ms\t"
        << "graph_search_time_ms\trerank_time_ms\ttotal_time_ms\thas_qrels\trelevant_doc_count\t"
        << "topk_relevant_hits\tfirst_relevant_rank\trecall\tmrr\thitrate\n";
}

void write_query_profile_row(std::ofstream& stream,
                             int query_idx,
                             int nprob,
                             int ef,
                             int rerank_k,
                             const QuerySearchProfile& profile,
                             bool has_qrels,
                             size_t relevant_doc_count,
                             size_t topk_relevant_hits,
                             int first_relevant_rank,
                             double recall,
                             double mrr,
                             double hitrate) {
    stream
        << query_idx << '\t'
        << profile.query_vecnum << '\t'
        << nprob << '\t'
        << ef << '\t'
        << rerank_k << '\t'
        << profile.selected_unique_cluster_count << '\t'
        << profile.entry_point_count << '\t'
        << join_int_vector(profile.selected_unique_cluster_ids) << '\t'
        << join_size_vector(profile.selected_unique_cluster_sizes) << '\t'
        << join_nested_int_vectors(profile.per_vector_clusters) << '\t'
        << join_nested_size_vectors(profile.per_vector_cluster_sizes) << '\t'
        << profile.merged_cluster_membership_sum << '\t'
        << profile.merged_routed_doc_union_size << '\t'
        << profile.graph_candidate_count << '\t'
        << profile.rerank_candidate_count << '\t'
        << profile.final_result_count << '\t'
        << (profile.routing_time_seconds * 1000.0) << '\t'
        << (profile.graph_search_time_seconds * 1000.0) << '\t'
        << (profile.rerank_time_seconds * 1000.0) << '\t'
        << (profile.total_time_seconds * 1000.0) << '\t'
        << (has_qrels ? 1 : 0) << '\t'
        << relevant_doc_count << '\t'
        << topk_relevant_hits << '\t'
        << first_relevant_rank << '\t'
        << recall << '\t'
        << mrr << '\t'
        << hitrate << '\n';
}


struct PaperShortcutPair {
    int qid;
    int positive_pid;
};

class Solution {
public:
    hnswlib::labeltype choose_cluster_entry_label(int cluster_id,
                                                  const std::vector<hnswlib::labeltype>& members) const {
        if (members.empty()) {
            return -1;
        }
        if (cluster_entry_mode == "random") {
            std::mt19937 rng(static_cast<uint32_t>(cluster_entry_seed + cluster_id * 1000003));
            std::uniform_int_distribution<size_t> dist(0, members.size() - 1);
            return members[dist(rng)];
        }
        return members[0];
    }

    void build_fine_cluster(int d, const std::vector<vectorset>& base, const std::vector<std::vector<hnswlib::labeltype>>& cluster_set, const std::vector<int>& temp, const float* cluster_distance) {
        double time = omp_get_wtime();
        alg_hnsw_list.resize(1);
        dimension = d;
        base_vectors = base;
        self_cluster_set = cluster_set;
        space_ptr = new hnswlib::L2VSSpace(dimension);
        temp_cluster_id = temp;
        std::cout << "init alg" << std::endl;
        cluster_entries.resize(NUM_GRAPH_CLUSTER);
        for (int i = 0; i < cluster_set.size(); i++) {
            cluster_entries[i] = -1;
        }
        alg_hnsw_list[0] = new hnswlib::HierarchicalNSW<float>(space_ptr, base.size() + 1, M_index, EF_index);
        for (int tmpi = 0; tmpi < temp_cluster_id.size(); tmpi++) {
            int i = temp_cluster_id[tmpi];
            if (cluster_set[i].empty()) {
                continue;
            }
            alg_hnsw_list[0]->setClusterAllowedLabels(cluster_set[i]);
            const hnswlib::labeltype entry_label = choose_cluster_entry_label(i, cluster_set[i]);
            cluster_entries[i] = entry_label;
            const double cur_time = omp_get_wtime();
            std::cout << "cluster build begin: " + std::to_string(i) + " " + std::to_string(cluster_set[i].size()) << std::endl;
            alg_hnsw_list[0]->addClusterPointEntry(&base_vectors[entry_label], cluster_distance, entry_label,
                                                   entry_label, nullptr);
            #pragma omp parallel for schedule(dynamic)
            for(int j = 0; j < static_cast<int>(cluster_set[i].size()); j++) {
                if (cluster_set[i][j] == entry_label) {
                    continue;
                }
                alg_hnsw_list[0]->addClusterPointEntry(&base_vectors[cluster_set[i][j]], cluster_distance,
                                                       cluster_set[i][j], entry_label, nullptr);
            }
            alg_hnsw_list[0]->clearClusterAllowedLabels();
            std::cout << "cluster build finish: " + std::to_string(i) + " " + std::to_string(omp_get_wtime() - cur_time) << std::endl;
        }
        alg_hnsw_list[0]->search_set.resize(alg_hnsw_list[0]->max_elements_);
        std::cout << "Build time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    double search_with_fine_cluster(const vectorset& query,
                                    std::vector<float>& query_cluster_scores,
                                    std::vector<float>& col_query_cluster_scores,
                                    std::vector<float>& center_data,
                                    std::vector<float>& graph_center_data,
                                    int k,
                                    int ef,
                                    std::vector<std::pair<int, float>>& res,
                                    QuerySearchProfile* profile = nullptr) {
        res.clear();
        res.resize(rerankK);

        alg_hnsw_list[0]->search_set.assign(alg_hnsw_list[0]->search_set.size(), 0);
        volatile int temp = 6;

        double start_time = omp_get_wtime();
        hnswlib::fast_dot_product_blas((&query)->vecnum, VECTOR_DIM, NUM_GRAPH_CLUSTER, (&query)->data, graph_center_data.data(), query_cluster_scores.data()); 
        std::unordered_set<int> unique_indices;
        std::vector<std::vector<int>> per_vector_clusters;
        if (profile != nullptr) {
            get_top_k_indices_col_with_details(query_cluster_scores, (&query)->vecnum, NUM_GRAPH_CLUSTER, NPROB,
                                               per_vector_clusters, unique_indices);
            profile->query_vecnum = query.vecnum;
            profile->per_vector_clusters = per_vector_clusters;
            profile->per_vector_cluster_sizes.resize(per_vector_clusters.size());
        } else {
            get_unique_top_k_indices_col(query_cluster_scores, (&query)->vecnum, NUM_GRAPH_CLUSTER, NPROB, unique_indices);
        }
        hnswlib::fast_dot_product_blas(NUM_CLUSTER, VECTOR_DIM, (&query)->vecnum, center_data.data(), (&query)->data, col_query_cluster_scores.data()); 
        vectorset query_cluster = vectorset(col_query_cluster_scores.data(), nullptr, NUM_CLUSTER, (&query)->vecnum);

        std::vector<hnswlib::labeltype> entry_points;
        std::vector<int> selected_unique_cluster_ids;
        std::vector<size_t> selected_unique_cluster_sizes;
        size_t merged_cluster_membership_sum = 0;
        
        for (const int idx : unique_indices) {
            selected_unique_cluster_ids.push_back(idx);
            selected_unique_cluster_sizes.push_back(self_cluster_set[idx].size());
            merged_cluster_membership_sum += self_cluster_set[idx].size();
            if (cluster_entries[idx] != -1) {
                entry_points.push_back(cluster_entries[idx]);
                for (const int j: self_cluster_set[idx]) {
                    alg_hnsw_list[0]->search_set[alg_hnsw_list[0]->label_lookup_[j]] = true;
                }
            }
        }
        if (profile != nullptr) {
            for (size_t row = 0; row < profile->per_vector_clusters.size(); ++row) {
                profile->per_vector_cluster_sizes[row].reserve(profile->per_vector_clusters[row].size());
                for (const int cluster_id : profile->per_vector_clusters[row]) {
                    profile->per_vector_cluster_sizes[row].push_back(self_cluster_set[cluster_id].size());
                }
            }
            std::vector<std::pair<int, size_t>> cluster_pairs;
            cluster_pairs.reserve(selected_unique_cluster_ids.size());
            for (size_t i = 0; i < selected_unique_cluster_ids.size(); ++i) {
                cluster_pairs.emplace_back(selected_unique_cluster_ids[i], selected_unique_cluster_sizes[i]);
            }
            std::sort(cluster_pairs.begin(), cluster_pairs.end(),
                      [](const std::pair<int, size_t>& a, const std::pair<int, size_t>& b) {
                          return a.first < b.first;
                      });
            profile->selected_unique_cluster_ids.clear();
            profile->selected_unique_cluster_sizes.clear();
            for (const auto& cluster_pair : cluster_pairs) {
                profile->selected_unique_cluster_ids.push_back(cluster_pair.first);
                profile->selected_unique_cluster_sizes.push_back(cluster_pair.second);
            }
            profile->selected_unique_cluster_count = cluster_pairs.size();
            profile->entry_point_count = entry_points.size();
            profile->merged_cluster_membership_sum = merged_cluster_membership_sum;
            size_t merged_routed_doc_union_size = 0;
            for (bool allowed : alg_hnsw_list[0]->search_set) {
                if (allowed) {
                    merged_routed_doc_union_size++;
                }
            }
            profile->merged_routed_doc_union_size = merged_routed_doc_union_size;
        }
        double cluster_time = omp_get_wtime();
        alg_hnsw_list[0]->setEf(ef);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_list[0]->searchKnnClusterEntries(&query_cluster, ef, entry_points);
        std::vector<std::pair<float, hnswlib::labeltype>> merge_result;
        while(result.size() > 0) {
            merge_result.push_back(result.top());
            result.pop();
        }

        double search_time = omp_get_wtime();
        if (profile != nullptr) {
            profile->graph_candidate_count = merge_result.size();
        }
        int numrerank = std::min((int)merge_result.size(), rerankK);
        std::partial_sort(merge_result.begin(), merge_result.begin() + numrerank, merge_result.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first < b.first;  // 按 float 排序，越小越靠前
                      });

        res.resize(numrerank);
        if (CPU_num == 1) {
            // using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_mat(query.data, query.vecnum, VECTOR_DIM);
            std::vector<float> decoded_doc_buffer;
            for (int i = 0; i < numrerank; i++){
                // for (const hnswlib::labeltype ind : search_result){
                hnswlib::labeltype ind = merge_result[i].second;
                const float* doc_data = materialize_vectorset_data(base_vectors[ind], decoded_doc_buffer);
                Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_mat(doc_data, base_vectors[ind].vecnum, VECTOR_DIM);
                Eigen::MatrixXf C = A_mat * B_mat.transpose();
                res[i] = std::make_pair(ind, 1 - C.rowwise().maxCoeff().sum() / query.vecnum);
            }            
        } else {
            std::vector<float> decoded_doc_buffer;
            for (int i = 0; i < numrerank; i++){
                // for (const hnswlib::labeltype ind : search_result){
                hnswlib::labeltype ind = merge_result[i].second;
                const float* doc_data = materialize_vectorset_data(base_vectors[ind], decoded_doc_buffer);
                vectorset decoded_doc(const_cast<float*>(doc_data), base_vectors[ind].dim, base_vectors[ind].vecnum);
                res[i] = std::make_pair(ind, hnswlib::L2SqrVecCF(&query, &decoded_doc, 0));
            }
        }

        std::sort(res.begin(), res.end(), 
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
            });
        if (res.size() > k) {
            res.resize(k);
        }

        double end_time = omp_get_wtime();
        if (profile != nullptr) {
            profile->rerank_candidate_count = numrerank;
            profile->final_result_count = res.size();
            profile->routing_time_seconds = cluster_time - start_time;
            profile->graph_search_time_seconds = search_time - cluster_time;
            profile->rerank_time_seconds = end_time - search_time;
            profile->total_time_seconds = end_time - start_time;
        }
        return end_time - start_time;
    }

    void save_fine_cluster(const std::string &location) {
        double time = omp_get_wtime();
        ensure_directory_exists(location);
        std::string locai = location + std::to_string(0) + ".bin";
        alg_hnsw_list[0]->saveIndex(locai);
        std::cout << "save time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    void load_fine_cluster(const std::string &location, int d, const std::vector<vectorset>& base, const std::vector<std::vector<hnswlib::labeltype>>& cluster_set, const std::vector<int>& temp) {
        double time = omp_get_wtime();
        self_cluster_set = cluster_set;
        dimension = d;
        base_vectors = base;
        space_ptr = new hnswlib::L2VSSpace(d);
        temp_cluster_id = temp;
        alg_hnsw_list.resize(1);
        cluster_entries.resize(NUM_GRAPH_CLUSTER);
        for (int i = 0; i < cluster_set.size(); i++) {
            // cluster_entries[i] = cluster_set[i][0];
            cluster_entries[i] = -1;
        }   

        alg_hnsw_list[0] = new hnswlib::HierarchicalNSW<float>(space_ptr, base.size() + 1, M_index, EF_index);
        alg_hnsw_list[0]->loadIndex(location + std::to_string(0) + ".bin", space_ptr);

        for(int tmpi = 0; tmpi < temp_cluster_id.size(); tmpi++) {
            int i = temp_cluster_id[tmpi];
            if (cluster_set[i].empty()) {
                continue;
            }
            cluster_entries[i] = choose_cluster_entry_label(i, cluster_set[i]);
            for (int j = 0; j < cluster_set[i].size(); j++) {
                alg_hnsw_list[0]->loadDataAddress(&base_vectors[cluster_set[i][j]], cluster_set[i][j]);
            }
            if (i % 1000 == 0) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;
        alg_hnsw_list[0]->search_set.resize(alg_hnsw_list[0]->max_elements_);
        std::cout << "load time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    void repair_fine_graph_structure(const std::vector<std::vector<hnswlib::labeltype>>& cluster_set) {
        double time = omp_get_wtime();
        for (int i = 0; i < cluster_set.size(); i++) {
            alg_hnsw_list[0]->entry_map.clear();
            // std::cout << "cluster: " << i << std::endl;
            if (i % 10000 == 0) {
                std::cout << i << std::endl;
            }
            if (cluster_set[i].empty()) {
                continue;
            }
            for (int d: cluster_set[i]) {
                alg_hnsw_list[0]->entry_map[alg_hnsw_list[0]->label_lookup_[d]] = i;
            }
            // std::cout << solution.alg_hnsw_list[0]->entry_map.size() << " " << cluster_set[i].size() << std::endl;
            std::vector<std::pair<hnswlib::tableint, int>>search_result = alg_hnsw_list[0]->searchNodesForFix(cluster_set[i][0], i);
            int l = 0;
            for (int j = 1; j < cluster_set[i].size(); j++) {
                int d = alg_hnsw_list[0]->label_lookup_[cluster_set[i][j]];
                bool connect_success = false;
                if (alg_hnsw_list[0]->entry_map[d] != i + 1) {
                    while(l < search_result.size()) {
                        if (alg_hnsw_list[0]->canAddEdgeinter(search_result[l].first)) {
                            alg_hnsw_list[0]->mutuallyConnectTwoInterElement(search_result[l].first, d);
                            if (alg_hnsw_list[0]->canAddEdgeinter(d)) {
                                alg_hnsw_list[0]->mutuallyConnectTwoInterElement(d, search_result[l].first);
                            }
                            connect_success = true;
                            break;
                        } else {
                            l++;
                        }
                    }
                    std::vector<std::pair<hnswlib::tableint, int>> newsearch_result = alg_hnsw_list[0]->searchNodesForFix(cluster_set[i][j], i);
                    for (int k = 0; k < newsearch_result.size(); k++) {
                        search_result.push_back(newsearch_result[k]);
                    }
                }
            }
            // std::cout << "cluster id: " << i << " doc id: " << corresponding_doc_id[tmpi] << " cluster size: " << cluster_set[i].size() << " searched size: " << search_result.size() << " hop distance: " << reach_hop << std::endl;
            // std::cout <<search_result.size() << " " << can_reach << std::endl;
            // std::cout << alg_hnsw_list[0]->searchNodes(cluster_set[i][0], i).size() << std::endl;
        }
        std::cout << "load time: " << omp_get_wtime() - time << "sec"<<std::endl;
    }

    void apply_shortcut_edges(const std::vector<std::pair<int, int>>& edge_pairs) {
        if (edge_pairs.empty()) {
            std::cout << "Shortcut edges: nothing to apply." << std::endl;
            return;
        }

        double time = omp_get_wtime();
        size_t missing_labels = 0;
        size_t full_neighbors = 0;
        size_t added_directed_edges = 0;
        size_t self_edges = 0;

        for (const auto& edge : edge_pairs) {
            if (edge.first == edge.second) {
                self_edges++;
                continue;
            }

            auto it1 = alg_hnsw_list[0]->label_lookup_.find(edge.first);
            auto it2 = alg_hnsw_list[0]->label_lookup_.find(edge.second);
            if (it1 == alg_hnsw_list[0]->label_lookup_.end() || it2 == alg_hnsw_list[0]->label_lookup_.end()) {
                missing_labels++;
                continue;
            }

            const hnswlib::tableint p1 = it1->second;
            const hnswlib::tableint p2 = it2->second;

            if (alg_hnsw_list[0]->canAddEdgeinter(p1)) {
                alg_hnsw_list[0]->mutuallyConnectTwoInterElement(p1, p2);
                added_directed_edges++;
            } else {
                full_neighbors++;
            }

            if (alg_hnsw_list[0]->canAddEdgeinter(p2)) {
                alg_hnsw_list[0]->mutuallyConnectTwoInterElement(p2, p1);
                added_directed_edges++;
            } else {
                full_neighbors++;
            }
        }

        std::cout << "Shortcut edges applied: input_pairs=" << edge_pairs.size()
                  << " directed_added=" << added_directed_edges
                  << " missing_labels=" << missing_labels
                  << " neighbor_lists_full=" << full_neighbors
                  << " self_edges_skipped=" << self_edges
                  << " time=" << (omp_get_wtime() - time) << "sec" << std::endl;
    }

    void apply_paper_shortcut_pairs(const std::vector<vectorset>& train_queries,
                                    const std::vector<PaperShortcutPair>& pairs,
                                    std::vector<float>& query_cluster_scores,
                                    std::vector<float>& col_query_cluster_scores,
                                    std::vector<float>& center_data,
                                    std::vector<float>& graph_center_data,
                                    int topf,
                                    int shortcut_ef_search,
                                    size_t degree_limit) {
        if (train_queries.empty() || pairs.empty()) {
            std::cout << "Paper shortcut injection: nothing to apply." << std::endl;
            return;
        }
        if (train_queries.size() != pairs.size()) {
            throw std::runtime_error("Paper shortcut query/pair size mismatch.");
        }

        double time = omp_get_wtime();
        size_t searched_pairs = 0;
        size_t already_found = 0;
        size_t missing_positive_labels = 0;
        size_t empty_results = 0;
        size_t degree_rejected = 0;
        size_t duplicate_edges = 0;
        size_t undirected_edges_added = 0;
        std::vector<std::pair<int, float>> search_result;
        const int saved_rerank = rerankK;
        rerankK = std::max(rerankK, topf);

        for (size_t i = 0; i < pairs.size(); ++i) {
            if (i % 1000 == 0) {
                std::cout << "paper shortcut progress: " << i << "/" << pairs.size() << std::endl;
            }
            search_result.clear();
            search_with_fine_cluster(train_queries[i], query_cluster_scores, col_query_cluster_scores,
                                     center_data, graph_center_data, topf, shortcut_ef_search, search_result);
            searched_pairs++;
            if (search_result.empty()) {
                empty_results++;
                continue;
            }

            const int positive_pid = pairs[i].positive_pid;
            bool found_positive = false;
            for (const auto& candidate : search_result) {
                if (candidate.first == positive_pid) {
                    found_positive = true;
                    break;
                }
            }
            if (found_positive) {
                already_found++;
                continue;
            }

            hnswlib::tableint positive_internal = 0;
            {
                std::unique_lock<std::mutex> lock(alg_hnsw_list[0]->label_lookup_lock);
                auto positive_it = alg_hnsw_list[0]->label_lookup_.find(positive_pid);
                if (positive_it == alg_hnsw_list[0]->label_lookup_.end()) {
                    missing_positive_labels++;
                    continue;
                }
                positive_internal = positive_it->second;
            }

            const int top_pid = search_result.front().first;
            if (top_pid == positive_pid) {
                already_found++;
                continue;
            }

            hnswlib::tableint top_internal = 0;
            {
                std::unique_lock<std::mutex> lock(alg_hnsw_list[0]->label_lookup_lock);
                auto top_it = alg_hnsw_list[0]->label_lookup_.find(top_pid);
                if (top_it == alg_hnsw_list[0]->label_lookup_.end()) {
                    missing_positive_labels++;
                    continue;
                }
                top_internal = top_it->second;
            }

            if (alg_hnsw_list[0]->hasLevel0Neighbor(top_internal, positive_internal) &&
                alg_hnsw_list[0]->hasLevel0Neighbor(positive_internal, top_internal)) {
                duplicate_edges++;
                continue;
            }

            if (!alg_hnsw_list[0]->canAddEdgeinterWithLimit(top_internal, degree_limit) ||
                !alg_hnsw_list[0]->canAddEdgeinterWithLimit(positive_internal, degree_limit)) {
                degree_rejected++;
                continue;
            }

            alg_hnsw_list[0]->mutuallyConnectTwoInterElement(top_internal, positive_internal);
            alg_hnsw_list[0]->mutuallyConnectTwoInterElement(positive_internal, top_internal);
            undirected_edges_added++;
        }

        rerankK = saved_rerank;
        std::cout << "Paper shortcut injection finish: searched_pairs=" << searched_pairs
                  << " already_found=" << already_found
                  << " missing_positive_labels=" << missing_positive_labels
                  << " empty_results=" << empty_results
                  << " degree_rejected=" << degree_rejected
                  << " duplicate_edges=" << duplicate_edges
                  << " undirected_edges_added=" << undirected_edges_added
                  << " degree_limit=" << degree_limit
                  << " topf=" << topf
                  << " ef_search=" << shortcut_ef_search
                  << " time=" << (omp_get_wtime() - time) << "sec" << std::endl;
    }

public:
    int dimension;
    std::vector<vectorset> base_vectors;
    hnswlib::L2VSSpace* space_ptr;
    // hnswlib::HierarchicalNSW<float>* alg_hnsw;
    std::vector<int> temp_cluster_id;
    std::vector<int> cluster_entries;
    std::vector<std::vector<hnswlib::labeltype>> self_cluster_set;
    std::vector<hnswlib::HierarchicalNSW<float>*> alg_hnsw_list;
    int cluster_entry_seed = 123;
    std::string cluster_entry_mode = "first";
};


float half_to_float(uint16_t h) {
    // 参考 IEEE 754 半精度转换公式
    int s = (h >> 15) & 0x1;                   // 符号位
    int e = (h >> 10) & 0x1F;                  // 指数部分
    int f = h & 0x3FF;                         // 尾数部分
    if (e == 0) {                              // 次正规数
        return (s ? -1 : 1) * std::ldexp(f, -24);
    } else if (e == 31) {                      // 特殊值（NaN 或 Infinity）
        return (s ? -1 : 1) * (f ? NAN : INFINITY);
    } else {                                   // 规范化数
        return (s ? -1 : 1) * std::ldexp(f + 1024, e - 15 - 10);
    }
}

uint16_t float_to_half_scalar(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }

    if (exponent >= 31) {
        if (mantissa != 0) {
            return static_cast<uint16_t>(sign | 0x7E00u);
        }
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x1000u) >> 13));
}

void decode_half_array_to_float(const uint16_t* src, float* dst, size_t count) {
#if defined(__F16C__)
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i half_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 float_vec = _mm256_cvtph_ps(half_vec);
        _mm256_storeu_ps(dst + i, float_vec);
    }
    for (; i < count; ++i) {
        dst[i] = half_to_float(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = half_to_float(src[i]);
    }
#endif
}

void encode_float_array_to_half(const float* src, uint16_t* dst, size_t count) {
#if defined(__F16C__)
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 float_vec = _mm256_loadu_ps(src + i);
        __m128i half_vec = _mm256_cvtps_ph(float_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), half_vec);
    }
    for (; i < count; ++i) {
        dst[i] = float_to_half_scalar(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = float_to_half_scalar(src[i]);
    }
#endif
}

void load_from_msmarco(std::vector<float>& base_data, std::vector<uint16_t>& base_data_half, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set,
                       int file_numbers, std::vector<std::vector<int>>& qrels,
                       bool use_full_precision_base) {
    long long offset = 0;
    long long all_elements = 0;
    long long code_offset = 0;
    long long all_codes = 0;
    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/qembs.npy";
    std::string qlensfile_name = dataset_path + "qdata/filterd_query_len.npy";
    std::string qrelfile_name = dataset_path + "qdata/qrels.tsv";

    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt";
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy";


    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/doc_codes_" + std::to_string(i) + ".npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        int64_t* lens_data = lens_npy.data<int64_t>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];

        if (use_full_precision_base) {
            decode_half_array_to_float(raw_vec_data, base_data.data() + all_elements, static_cast<size_t>(num_elements));
        } else {
            std::memcpy(base_data_half.data() + all_elements,
                        raw_vec_data,
                        static_cast<size_t>(num_elements) * sizeof(uint16_t));
        }
        for (long long j = 0; j < static_cast<long long>(num_codes); ++j) {
            base_data_codes[all_codes + j] = static_cast<int>(codes_data[j]);
        }

        all_elements += num_elements;
        all_codes += num_codes;

        for (size_t j = 0; j < doc_num; ++j) {
            if (use_full_precision_base) {
                base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, static_cast<int>(lens_data[j])));
            } else {
                base.push_back(vectorset(base_data_half.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, static_cast<int>(lens_data[j])));
            }
            offset += lens_data[j] * VECTOR_DIM;
            code_offset += lens_data[j];
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = static_cast<float>(half_to_float(raw_cembs_data[i]));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = static_cast<float>(raw_gcembs_data[i]);
    }

    int q_num = 0;
    if (file_exists(qembfile_name)) {
        cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
        float* raw_qembs_data = qembs_npy.data<float>();
        q_num = static_cast<int>(qembs_npy.shape[0]);
        size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1] * qembs_npy.shape[2];

        int q_offset = 0;
        for (size_t i = 0; i < num_qembs_elements; ++i) {
            query_data[i] = static_cast<float>(raw_qembs_data[i]);
        }

        for (int i = 0; i < q_num; ++i) {
            query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, QUERY_VECTOR_COUNT));
            q_offset += QUERY_VECTOR_COUNT * VECTOR_DIM;
        }
    } else {
        qembfile_name = dataset_path + "qdata/filterd_query.npy";
        if (!file_exists(qembfile_name) || !file_exists(qlensfile_name)) {
            throw std::runtime_error("Missing query embeddings for dataset root: " + dataset_path);
        }

        cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
        cnpy::NpyArray qlens_npy = cnpy::npy_load(qlensfile_name);
        float* raw_qembs_data = qembs_npy.data<float>();
        const int64_t* qlens_data = qlens_npy.data<int64_t>();
        q_num = static_cast<int>(qlens_npy.shape[0]);
        size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1];

        long long q_offset = 0;
        for (size_t i = 0; i < num_qembs_elements; ++i) {
            query_data[i] = static_cast<float>(raw_qembs_data[i]);
        }

        for (int i = 0; i < q_num; ++i) {
            const int qlen = static_cast<int>(qlens_data[i]);
            query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, qlen));
            q_offset += static_cast<long long>(qlen) * VECTOR_DIM;
        }
    }
    load_qrels_if_exists(qrelfile_name, q_num, qrels);

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) {
        std::istringstream iss(cdocs_line);
        hnswlib::labeltype num1;
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        lineid++;
    }
    codcsfile.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}

void load_from_okvqa(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0; 
    long long code_offset = 0;
    long long all_codes = 0;  

    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/filterd_query.npy";
    std::string qrelfile_name = dataset_path + "qdata/qrels.tsv"; 
    std::string qlensfile_name = dataset_path + "qdata/filterd_query_len.npy";

    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt"; 
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy"; 

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/" + std::to_string(i) + ".codes.npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";

        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];
        // std::cout << codes_npy.word_size << std::endl;
        // std::cout << sizeof(int32_t) << std::endl;
        // std::cout << sizeof(int16_t) << std::endl;
        // std::cout << sizeof(int) << std::endl;
        std::cout << num_codes << std::endl;
        std::cout << all_codes << std::endl;
        std::cout << "Processing file " << i << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        // std::cout << "?" << std::endl;
        for (long long i = 0; i < num_codes; ++i) {
            base_data_codes[all_codes + i] = static_cast<int>(codes_data[i]);
            if (i < 10) {
                // std::cout << i << " " << all_codes + i << std::endl;
                std::cout << codes_data[i] << " ";
                std::cout << base_data_codes[all_codes + i] << " ";
            }
        }
        std::cout << std::endl;

        all_elements += num_elements;
        all_codes += num_codes;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
            code_offset += lens_data[i].real();
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = (static_cast<float>(half_to_float(raw_cembs_data[i])));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = (static_cast<float>((raw_gcembs_data[i])));
    }
    // graph_center_data = center_data;
    // std::cout << num_cembs_elements << std::endl;

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    cnpy::NpyArray qlens_npy = cnpy::npy_load(qlensfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1];
    size_t q_num = NUM_QUERT_OKVQA;

    std::complex<int>* qlens_data = qlens_npy.data<std::complex<int>>();

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }

    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, qlens_data[i].real()));
        q_offset += qlens_data[i].real() * VECTOR_DIM;
    }

    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
        std::istringstream iss(cdocs_line);  // 创建字符串流
        hnswlib::labeltype num1;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        if (lineid % 100 == 0) {
            std::cout << lineid << " " << cluster_set[lineid][cluster_set[lineid].size()-1] << std::endl;
        }
        lineid++;
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}


void load_from_evqa(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0; 
    long long code_offset = 0;
    long long all_codes = 0;  

    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/filterd_query.npy";
    std::string qrelfile_name = dataset_path + "qdata/qrels.tsv"; 
    std::string qlensfile_name = dataset_path + "qdata/filterd_query_len.npy";
    
    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt"; 
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy"; 

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/" + std::to_string(i) + ".codes.npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];
        // std::cout << codes_npy.word_size << std::endl;
        // std::cout << sizeof(int32_t) << std::endl;
        // std::cout << sizeof(int16_t) << std::endl;
        // std::cout << sizeof(int) << std::endl;
        std::cout << num_codes << std::endl;
        std::cout << all_codes << std::endl;
        std::cout << "Processing file " << i << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        // std::cout << "?" << std::endl;
        for (long long i = 0; i < num_codes; ++i) {
            base_data_codes[all_codes + i] = static_cast<int>(codes_data[i]);
            if (i < 10) {
                // std::cout << i << " " << all_codes + i << std::endl;
                std::cout << codes_data[i] << " ";
                std::cout << base_data_codes[all_codes + i] << " ";
            }
        }
        std::cout << std::endl;

        all_elements += num_elements;
        all_codes += num_codes;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
            code_offset += lens_data[i].real();
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = (static_cast<float>(half_to_float(raw_cembs_data[i])));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = (static_cast<float>((raw_gcembs_data[i])));
    }
    // graph_center_data = center_data;
    // std::cout << num_cembs_elements << std::endl;

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    cnpy::NpyArray qlens_npy = cnpy::npy_load(qlensfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1];
    size_t q_num = NUM_QUERT_EVQA;

    std::complex<int>* qlens_data = qlens_npy.data<std::complex<int>>();

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }

    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, qlens_data[i].real()));
        q_offset += qlens_data[i].real() * VECTOR_DIM;
    }

    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
        std::istringstream iss(cdocs_line);  // 创建字符串流
        hnswlib::labeltype num1;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        if (lineid % 100 == 0) {
            std::cout << lineid << " " << cluster_set[lineid][cluster_set[lineid].size()-1] << std::endl;
        }
        lineid++;
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}

void load_from_scidocs(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set,
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;
    long long all_elements = 0;
    long long code_offset = 0;
    long long all_codes = 0;

    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/filterd_query.npy";
    std::string qrelfile_name = dataset_path + "qdata/qrels.tsv";
    std::string qlensfile_name = dataset_path + "qdata/filterd_query_len.npy";
    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt";
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy";

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/doc_codes_" + std::to_string(i) + ".npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);

        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];
        int64_t* lens_data = lens_npy.data<int64_t>();
        size_t doc_num = lens_npy.shape[0];
        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];

        for (long long j = 0; j < static_cast<long long>(num_elements); ++j) {
            base_data[all_elements + j] = static_cast<float>(half_to_float(raw_vec_data[j]));
        }
        for (long long j = 0; j < static_cast<long long>(num_codes); ++j) {
            base_data_codes[all_codes + j] = static_cast<int>(codes_data[j]);
        }

        all_elements += num_elements;
        all_codes += num_codes;

        for (size_t j = 0; j < doc_num; ++j) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, static_cast<int>(lens_data[j])));
            offset += lens_data[j] * VECTOR_DIM;
            code_offset += lens_data[j];
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = static_cast<float>(half_to_float(raw_cembs_data[i]));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = static_cast<float>(raw_gcembs_data[i]);
    }

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    cnpy::NpyArray qlens_npy = cnpy::npy_load(qlensfile_name);
    float* raw_qembs_data = qembs_npy.data<float>();
    size_t num_qembs_elements = qembs_npy.shape[0] * qembs_npy.shape[1];
    size_t q_num = qlens_npy.shape[0];
    int64_t* qlens_data = qlens_npy.data<int64_t>();

    int q_offset = 0;
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = static_cast<float>(raw_qembs_data[i]);
    }
    for (size_t i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, static_cast<int>(qlens_data[i])));
        q_offset += qlens_data[i] * VECTOR_DIM;
    }

    load_qrels_if_exists(qrelfile_name, q_num, qrels);

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) {
        std::istringstream iss(cdocs_line);
        hnswlib::labeltype num1;
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        lineid++;
    }
    codcsfile.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}

void load_from_lotte(std::vector<float>& base_data, std::vector<vectorset>& base,
                       std::vector<float>& query_data, std::vector<vectorset>& query,
                       std::vector<int>& base_data_codes, std::vector<float>& center_data,
                       std::vector<float>& graph_center_data,
                       std::vector<std::vector<hnswlib::labeltype>>& cluster_set, 
                       int file_numbers, std::vector<std::vector<int>>& qrels) {
    long long offset = 0;  
    long long all_elements = 0; 
    long long code_offset = 0;
    long long all_codes = 0;  

    std::string cembfile_name = dataset_path + "cdata/centroids.npy";
    std::string qembfile_name = dataset_path + "qdata/lotte_pooled_dev_query.npy";
    std::string qrelfile_name = dataset_path + "qdata/qas.search.tsv"; 

    std::string cdocsfile_name = dataset_path + "cdata/coarse_cluster_info.txt"; 
    std::string gcembfile_name = dataset_path + "cdata/coarse_centroids.npy"; 

    for (int i = 0; i < file_numbers; i++) {
        std::string embfile_name = dataset_path + "docdata/encoding" + std::to_string(i) + "_float16.npy";
        std::string codesfile_name = dataset_path + "docdata/doc_codes_" + std::to_string(i) + ".npy";
        std::string lensfile_name = dataset_path + "docdata/doclens" + std::to_string(i) + ".npy";
        cnpy::NpyArray arr_npy = cnpy::npy_load(embfile_name);
        cnpy::NpyArray codes_npy = cnpy::npy_load(codesfile_name);
        cnpy::NpyArray lens_npy = cnpy::npy_load(lensfile_name);
        uint16_t* raw_vec_data = arr_npy.data<uint16_t>();
        size_t num_elements = arr_npy.shape[0] * arr_npy.shape[1];

        std::complex<int>* lens_data = lens_npy.data<std::complex<int>>();
        size_t doc_num = lens_npy.shape[0];

        int32_t* codes_data = codes_npy.data<int32_t>();
        size_t num_codes = codes_npy.shape[0];
        // std::cout << codes_npy.word_size << std::endl;
        // std::cout << sizeof(int32_t) << std::endl;
        // std::cout << sizeof(int16_t) << std::endl;
        // std::cout << sizeof(int) << std::endl;
        std::cout << num_codes << std::endl;
        std::cout << all_codes << std::endl;
        std::cout << "Processing file " << i << std::endl;
        
        for (long long i = 0; i < num_elements; ++i) {
            base_data[all_elements + i] = (static_cast<float>(half_to_float(raw_vec_data[i])));
        }
        // std::cout << "?" << std::endl;
        for (long long i = 0; i < num_codes; ++i) {
            base_data_codes[all_codes + i] = static_cast<int>(codes_data[i]);
            if (i < 10) {
                // std::cout << i << " " << all_codes + i << std::endl;
                std::cout << codes_data[i] << " ";
                std::cout << base_data_codes[all_codes + i] << " ";
            }
        }
        std::cout << std::endl;

        all_elements += num_elements;
        all_codes += num_codes;
        
        for (int i = 0; i < doc_num; ++i) {
            base.push_back(vectorset(base_data.data() + offset, base_data_codes.data() + code_offset, VECTOR_DIM, lens_data[i].real()));
            offset += lens_data[i].real() * VECTOR_DIM;
            code_offset += lens_data[i].real();
        }
    }

    cnpy::NpyArray cembs_npy = cnpy::npy_load(cembfile_name);
    uint16_t* raw_cembs_data = cembs_npy.data<uint16_t>();
    size_t num_cembs_elements = cembs_npy.shape[0] * cembs_npy.shape[1];
    for (size_t i = 0; i < num_cembs_elements; ++i) {
        center_data[i] = (static_cast<float>(half_to_float(raw_cembs_data[i])));
    }

    cnpy::NpyArray gcembs_npy = cnpy::npy_load(gcembfile_name);
    float* raw_gcembs_data = gcembs_npy.data<float>();
    size_t num_gcembs_elements = gcembs_npy.shape[0] * gcembs_npy.shape[1];
    for (size_t i = 0; i < num_gcembs_elements; ++i) {
        graph_center_data[i] = (static_cast<float>((raw_gcembs_data[i])));
    }
    // graph_center_data = center_data;
    // std::cout << num_cembs_elements << std::endl;

    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);

    float* raw_qembs_data = qembs_npy.data<float>();
    size_t q_num = qembs_npy.shape[0];
    size_t num_qembs_elements = q_num * qembs_npy.shape[1] * qembs_npy.shape[2];

    int q_offset = 0;
    
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = (static_cast<float>((raw_qembs_data[i])));
    }
    
    for (int i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, nullptr, VECTOR_DIM, QUERY_VECTOR_COUNT));
        q_offset += QUERY_VECTOR_COUNT * VECTOR_DIM;
    }
    qrels.resize(q_num + 1);

    std::ifstream file(qrelfile_name);
    std::string line;
    while (std::getline(file, line)) { // 逐行读取
        std::istringstream iss(line);  // 创建字符串流
        int num1, num2;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        if (iss >> num1 >> num2) {
            if (num1 < 0 || num1 >= q_num) {
                continue;
                // std::cerr << "?" << line << std::endl;
            } else {
                // std::cout << num1 << " " << num2 << std::endl;
                qrels[num1].push_back(num2);
            }
        }
    }
    file.close();

    std::ifstream codcsfile(cdocsfile_name);
    std::string cdocs_line;
    int lineid = 0;
    while (std::getline(codcsfile, cdocs_line)) { // 逐行读取
        std::istringstream iss(cdocs_line);  // 创建字符串流
        hnswlib::labeltype num1;
        char delimiter;                // 用于捕获 \t 分隔符

        // 读取两个整数，用 \t 作为分隔符
        while (iss >> num1) {
            cluster_set[lineid].push_back(num1);
        }
        if (lineid % 100 == 0) {
            std::cout << lineid << " " << cluster_set[lineid][cluster_set[lineid].size()-1] << std::endl;
        }
        lineid++;
    }
    file.close();

    std::cout << "load data finish! passage count: " << base.size() << " query count: " << query.size() << " " << qrels.size() << std::endl;
}

void load_pair_aligned_query_pairs(const std::string& qembfile_name,
                                   const std::string& qrelfile_name,
                                   std::vector<float>& query_data,
                                   std::vector<vectorset>& query,
                                   std::vector<PaperShortcutPair>& pairs) {
    cnpy::NpyArray qembs_npy = cnpy::npy_load(qembfile_name);
    if (qembs_npy.shape.size() != 3) {
        throw std::runtime_error("Expected pair-aligned query embeddings with shape [N, L, D]: " + qembfile_name);
    }

    const size_t q_num = qembs_npy.shape[0];
    const size_t query_vecnum = qembs_npy.shape[1];
    const size_t query_dim = qembs_npy.shape[2];
    if (static_cast<int>(query_dim) != VECTOR_DIM) {
        throw std::runtime_error("Pair query dim mismatch: expected " + std::to_string(VECTOR_DIM) +
                                 " got " + std::to_string(query_dim) + " from " + qembfile_name);
    }
    if (static_cast<int>(query_vecnum) != QUERY_VECTOR_COUNT) {
        throw std::runtime_error("Pair query vector count mismatch: expected " + std::to_string(QUERY_VECTOR_COUNT) +
                                 " got " + std::to_string(query_vecnum) + " from " + qembfile_name);
    }

    const size_t num_qembs_elements = q_num * query_vecnum * query_dim;
    query_data.resize(num_qembs_elements);
    const float* raw_qembs_data = qembs_npy.data<float>();
    for (size_t i = 0; i < num_qembs_elements; ++i) {
        query_data[i] = raw_qembs_data[i];
    }

    query.clear();
    query.reserve(q_num);
    size_t q_offset = 0;
    for (size_t i = 0; i < q_num; ++i) {
        query.push_back(vectorset(query_data.data() + q_offset, VECTOR_DIM, static_cast<int>(query_vecnum)));
        q_offset += query_vecnum * query_dim;
    }

    pairs.clear();
    pairs.reserve(q_num);
    std::ifstream file(qrelfile_name);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open pair qrels file: " + qrelfile_name);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        int qid = -1;
        int pid = -1;
        int score = 0;
        if (iss >> qid >> pid) {
            if (qid == -1 || pid == -1) {
                continue;
            }
            if (qid == 0 && pid == 0 && pairs.empty()) {
                continue;
            }
            pairs.push_back({qid, pid});
        }
    }
    file.close();

    if (pairs.size() != q_num) {
        throw std::runtime_error("Pair query/qrels size mismatch: " + std::to_string(q_num) +
                                 " embeddings vs " + std::to_string(pairs.size()) +
                                 " qrels rows for " + qembfile_name + " and " + qrelfile_name);
    }

    std::cout << "load pair-aligned shortcut queries finish! query count: " << query.size()
              << " pair count: " << pairs.size() << std::endl;
}

bool load_edge_pairs_from_file(const std::string& edgefile_name, std::vector<std::pair<int, int>>& edge_pair) {
    if (edgefile_name.empty()) {
        return false;
    }
    std::ifstream file(edgefile_name);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shortcut edge file: " + edgefile_name);
    }
    std::string line;
    size_t skipped_lines = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream iss(line);
        int num1 = -1;
        int num2 = -1;
        if (iss >> num1 >> num2) {
            if (num1 != num2) {
                edge_pair.push_back(std::make_pair(num1, num2));
            }
        } else {
            skipped_lines++;
        }
    }
    file.close();
    std::cout << "load shortcut edge finish! path: " << edgefile_name
              << " edge count: " << edge_pair.size()
              << " skipped lines: " << skipped_lines << std::endl;
    return true;
}


double calculate_recall_for_datasetlabel(const std::vector<std::pair<int, float>>& solution_indices,
                        const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    for (const auto& pair : solution_indices) {
        solution_set.insert(pair.first);
        // if (solution_set.size() >= K) {
        //     break;
        // }
    }
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
    }
    int intersection_count = 0;
    for (const int& index : solution_set) {
        if (ground_truth_set.find(index) != ground_truth_set.end()) {
            intersection_count++;
        }
    }

    double recall = static_cast<double>(intersection_count) / ground_truth_set.size();
    // double recall = static_cast<double>(intersection_count > 0);
    return recall;
}

double calculate_hitrate_for_datasetlabel(const std::vector<std::pair<int, float>>& solution_indices,
                        const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> solution_set;
    std::unordered_set<int> ground_truth_set;
    for (const auto& pair : solution_indices) {
        solution_set.insert(pair.first);
    }
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
    }
    int intersection_count = 0;
    for (const int& index : solution_set) {
        if (ground_truth_set.find(index) != ground_truth_set.end()) {
            intersection_count++;
        }
    }
    // double hitrate = static_cast<double>(intersection_count) / ground_truth_set.size();
    double hitrate = static_cast<double>(intersection_count > 0);
    return hitrate;
}

double calculate_mrr_for_datasetlabel(const std::vector<std::pair<int, float>>& solution_indices,
    const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> ground_truth_set;
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
    }
    const size_t limit = std::min(solution_indices.size(), static_cast<size_t>(K));
    for (size_t rank = 0; rank < limit; ++rank) {
        int pid = solution_indices[rank].first;
        if (ground_truth_set.find(pid) != ground_truth_set.end()) {
            // MRR: Reciprocal of (1-based) rank
            return 1.0 / (rank + 1);
        }
    }
    // No relevant document found
    return 0.0;
}

int calculate_first_hit_rank_for_datasetlabel(const std::vector<std::pair<int, float>>& solution_indices,
                                              const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> ground_truth_set;
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
    }
    const size_t limit = std::min(solution_indices.size(), static_cast<size_t>(K));
    for (size_t rank = 0; rank < limit; ++rank) {
        if (ground_truth_set.find(solution_indices[rank].first) != ground_truth_set.end()) {
            return static_cast<int>(rank + 1);
        }
    }
    return -1;
}

size_t calculate_hit_count_for_datasetlabel(const std::vector<std::pair<int, float>>& solution_indices,
                                            const std::vector<int>& ground_truth_indices) {
    std::unordered_set<int> ground_truth_set;
    for (const auto& pid : ground_truth_indices) {
        ground_truth_set.insert(pid);
    }
    size_t hit_count = 0;
    const size_t limit = std::min(solution_indices.size(), static_cast<size_t>(K));
    for (size_t rank = 0; rank < limit; ++rank) {
        if (ground_truth_set.find(solution_indices[rank].first) != ground_truth_set.end()) {
            hit_count++;
        }
    }
    return hit_count;
}

const float* materialize_vectorset_data(const vectorset& vectors,
                                        std::vector<float>& decoded_buffer) {
    if (!vectors.data_is_half) {
        return vectors.data;
    }
    const size_t total_values = vectors.vecnum * vectors.dim;
    if (decoded_buffer.size() < total_values) {
        decoded_buffer.resize(total_values);
    }
    decode_half_array_to_float(vectors.data_half, decoded_buffer.data(), total_values);
    return decoded_buffer.data();
}

int main() {
    install_fatal_signal_handlers();
    try {
    omp_set_nested(1);

    std::vector<float> base_data;
    std::vector<uint16_t> base_data_half;
    std::vector<int> base_vec_num;
    std::vector<float> query_data;
    std::vector<float> shortcut_query_data;
    std::vector<vectorset> base;
    std::vector<vectorset> query;
    std::vector<vectorset> shortcut_train_query;
    std::vector<int> base_data_codes;
    std::vector<float> center_data;
    std::vector<float> graph_center_data;
    std::vector<float> query_cluster_scores;
    std::vector<float> test_query_cluster_scores;
    std::vector<std::vector<hnswlib::labeltype>> cluster_set;
    std::vector<std::vector<int>> qrels;
    std::vector<PaperShortcutPair> shortcut_pairs;

    const std::string dataset_mode = get_env_string_or_default("GEM_DATASET", "msmarco");
    if (dataset_mode == "msmarco") {
        dataset = TEST_MSMARCO;
    } else if (dataset_mode == "scidocs") {
        dataset = TEST_SCIDOCS;
    } else if (dataset_mode == "generic" || dataset_mode == "nq" || dataset_mode == "fiqa") {
        dataset = TEST_GENERIC;
    } else {
        throw std::runtime_error("Unsupported GEM_DATASET: " + dataset_mode);
    }

    EF_index = get_env_int_or_default("GEM_EF_INDEX", EF_index);
    bool rebuild = get_env_bool_or_default("GEM_REBUILD", false);
    bool save_result = true;
    bool skip_search = get_env_bool_or_default("GEM_SKIP_SEARCH", false);
    bool query_profile_enabled = get_env_bool_or_default("GEM_QUERY_PROFILE", false);
    int build_thread_count = get_env_int_or_default("GEM_BUILD_THREADS", 0);
    int search_thread_count = get_env_int_or_default("GEM_SEARCH_THREADS", 1);
    int query_limit = get_env_int_or_default("GEM_QUERY_LIMIT", 0);
    int cluster_distance_block_rows = get_env_int_or_default("GEM_CLUSTER_DISTANCE_BLOCK_ROWS", 1024);
    bool msmarco_base_fp32 = get_env_bool_or_default("GEM_MSMARCO_BASE_FP32", false);
    bool apply_repair_on_build = get_env_bool_or_default("GEM_APPLY_REPAIR_ON_BUILD", false);
    bool apply_repair_on_load = get_env_bool_or_default("GEM_APPLY_REPAIR_ON_LOAD", false);
    int cluster_entry_seed = get_env_int_or_default("GEM_CLUSTER_ENTRY_SEED", 123);
    const std::string cluster_entry_mode = get_env_string_or_default("GEM_CLUSTER_ENTRY_MODE", "first");
    int shortcut_topf = get_env_int_or_default("GEM_SHORTCUT_TOPF", 100);
    int shortcut_ef_search = get_env_int_or_default("GEM_PAPER_SHORTCUT_EF_SEARCH", shortcut_topf);
    eflist = get_env_int_list_or_default("GEM_EF_LIST", eflist);
    std::vector<int> reranklist = get_env_int_list_or_default("GEM_RERANK_LIST", {128, 256, 378, 512});
    std::vector<int> nproblist = get_env_int_list_or_default(
        "GEM_NPROB_LIST",
        {get_env_int_or_default("GEM_NPROB", NPROB)});
    const std::string index_root_override = get_env_string_or_default("GEM_INDEX_ROOT", "/data1/ali/msmarco-gem-data/example_index");
    const std::string result_dir_override = get_env_string_or_default("GEM_RESULTS_DIR", "");
    const std::string msmarco_dataset_root_override = get_env_string_or_default("GEM_MSMARCO_DATASET_PATH", "/data/ali/msmarco-gem-data/");
    const std::string generic_dataset_root_override = get_env_string_or_default("GEM_DATASET_PATH", "");
    const std::string generic_dataset_name = get_env_string_or_default("GEM_DATASET_NAME", dataset_mode);
    const int generic_num_cluster_override = get_env_int_or_default("GEM_NUM_CLUSTER", 0);
    const int generic_num_graph_cluster_override = get_env_int_or_default("GEM_NUM_GRAPH_CLUSTER", 0);
    const std::string shortcut_mode = get_env_string_or_default("GEM_SHORTCUT_MODE", "");
    const std::string shortcut_edge_file = get_env_string_or_default("GEM_SHORTCUT_EDGE_FILE", "");
    const std::string paper_shortcut_train_qembs = get_env_string_or_default("GEM_PAPER_SHORTCUT_TRAIN_QEMBS", "");
    const std::string paper_shortcut_train_qrels = get_env_string_or_default("GEM_PAPER_SHORTCUT_TRAIN_QRELS", "");

    std::string index_file, save_result_file;

    std::vector<int> temp_cluster_id;

    if (dataset == 0) {
        NUM_BASE_SETS = NUM_BASE_SETS_MS;
        NUM_QUERY_SETS = NUM_QUERT_MS;
        NUM_CLUSTER = NUM_CLUSTER_MS;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_MS;
        dataset_path = msmarco_dataset_root_override;
        if (!dataset_path.empty() && dataset_path.back() != '/') {
            dataset_path.push_back('/');
        }
        VECTOR_DIM = infer_vector_dim(dataset_path + "docdata/encoding0_float16.npy", dataset_path + "qdata/qembs.npy");
        {
            const std::pair<int, int> query_layout = infer_query_layout(dataset_path + "qdata/qembs.npy");
            NUM_QUERY_SETS = query_layout.first;
            QUERY_VECTOR_COUNT = query_layout.second;
        }
        const long long num_base_vectors = infer_total_vectors_from_shards(dataset_path + "docdata/", MSMACRO_TEST_NUMBER);
        NUM_BASE_SETS = infer_total_docs_from_shards(dataset_path + "docdata/", MSMACRO_TEST_NUMBER);
        index_file = index_root_override + "/msmarcoIndex" + std::to_string(NUM_GRAPH_CLUSTER_MS) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        if (!result_dir_override.empty()) {
            save_result_file = result_dir_override + "/msmarco_results_" + std::to_string(NUM_GRAPH_CLUSTER_MS) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
        } else {
            save_result_file = index_root_override + "/msmarco_results_" + std::to_string(NUM_GRAPH_CLUSTER_MS) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
        }
        std::cout << index_file << std::endl;
        std::cout << "Using MSMARCO dataset root: " << dataset_path << std::endl;
        std::cout << "Using vector dimension: " << VECTOR_DIM << std::endl;
        std::cout << "Using query count/vector count: " << NUM_QUERY_SETS << " / " << QUERY_VECTOR_COUNT << std::endl;
        std::cout << "Using base set/vector count: " << NUM_BASE_SETS << " / " << num_base_vectors << std::endl;
        // test on all msmacro dataset
        if (msmarco_base_fp32) {
            base_data.resize(num_base_vectors * VECTOR_DIM);
        } else {
            base_data_half.resize(num_base_vectors * VECTOR_DIM);
        }
        query_data.resize((long long) get_npy_element_count(dataset_path + "qdata/qembs.npy"));
        base_data_codes.resize(num_base_vectors);
        center_data.resize((long long) NUM_CLUSTER * VECTOR_DIM);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * VECTOR_DIM);
        // cluster_set.resize(NUM_GRAPH_CLUSTER);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        // std::cout<< (long long) 25000 * MSMACRO_TEST_NUMBER * 80 << std::endl;
        std::cout << "MSMARCO base storage: " << (msmarco_base_fp32 ? "fp32" : "fp16") << " in RAM" << std::endl;
        load_from_msmarco(base_data, base_data_half, base, query_data, query, base_data_codes, center_data, graph_center_data,
                          cluster_set, MSMACRO_TEST_NUMBER, qrels, msmarco_base_fp32);
    }
    else if (dataset == 1) {
        NUM_BASE_SETS = NUM_BASE_SETS_LOTTE;
        NUM_QUERY_SETS = NUM_QUERT_LOTTE;
        NUM_CLUSTER = NUM_CLUSTER_LOTTE;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_LOTTE;
        dataset_path = dataset_path + "lotte/";
        VECTOR_DIM = infer_vector_dim(dataset_path + "docdata/encoding0_float16.npy", dataset_path + "qdata/lotte_pooled_dev_query.npy");
        {
            const std::pair<int, int> query_layout = infer_query_layout(dataset_path + "qdata/lotte_pooled_dev_query.npy");
            NUM_QUERY_SETS = query_layout.first;
            QUERY_VECTOR_COUNT = query_layout.second;
        }
        std::cout << "Using vector dimension: " << VECTOR_DIM << std::endl;
        std::cout << "Using query count/vector count: " << NUM_QUERY_SETS << " / " << QUERY_VECTOR_COUNT << std::endl;
        base_data.resize((long long) NUM_BASE_VECTOR_LOTTE * VECTOR_DIM);
        query_data.resize((long long) NUM_QUERY_SETS * VECTOR_DIM * QUERY_VECTOR_COUNT);
        base_data_codes.resize((long long) NUM_BASE_VECTOR_LOTTE);
        center_data.resize((long long) NUM_CLUSTER * VECTOR_DIM);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * VECTOR_DIM);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        load_from_lotte(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, LOTTE_TEST_NUMBER, qrels);
        index_file = "../../example_index/lotteIndex" + std::to_string(NUM_GRAPH_CLUSTER_LOTTE) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        save_result_file = "../../example_index/lotte_results_" + std::to_string(NUM_GRAPH_CLUSTER_LOTTE) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
    } else if (dataset == 2) {
        // OKVQA
        NUM_BASE_SETS = NUM_BASE_SETS_OKVQA;
        NUM_QUERY_SETS = NUM_QUERT_OKVQA;
        NUM_CLUSTER = NUM_CLUSTER_OKVQA;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_OKVQA;
        QUERY_VECTOR_COUNT = 320;
        dataset_path = dataset_path + "okvqa/";
        VECTOR_DIM = infer_vector_dim(dataset_path + "docdata/encoding0_float16.npy", dataset_path + "qdata/filterd_query.npy");
        std::cout << "Using vector dimension: " << VECTOR_DIM << std::endl;
        base_data.resize((long long) NUM_BASE_VECTOR_OKVQA * VECTOR_DIM);
        query_data.resize((long long) NUM_QUERT_OKVQA * VECTOR_DIM * QUERY_VECTOR_COUNT);
        base_data_codes.resize((long long) NUM_BASE_VECTOR_OKVQA);
        center_data.resize((long long) NUM_CLUSTER * VECTOR_DIM);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * VECTOR_DIM);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        load_from_okvqa(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, OKVQA_TEST_NUMBER, qrels);
        index_file = "../../example_index/okvqaIndex" + std::to_string(NUM_GRAPH_CLUSTER_OKVQA) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        save_result_file = "../../example_index/okvqa_results_" + std::to_string(NUM_GRAPH_CLUSTER_OKVQA) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
    } else if (dataset == 3) {
        // OKVQA
        NUM_BASE_SETS = NUM_BASE_SETS_EVQA;
        NUM_QUERY_SETS = NUM_QUERT_EVQA;
        NUM_CLUSTER = NUM_CLUSTER_EVQA;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_EVQA;
        QUERY_VECTOR_COUNT = 320;
        dataset_path = dataset_path + "evqa/"; 
        VECTOR_DIM = infer_vector_dim(dataset_path + "docdata/encoding0_float16.npy", dataset_path + "qdata/filterd_query.npy");
        std::cout << "Using vector dimension: " << VECTOR_DIM << std::endl;
        base_data.resize((long long) NUM_BASE_VECTOR_EVQA * VECTOR_DIM);
        query_data.resize((long long) NUM_QUERT_EVQA * VECTOR_DIM * QUERY_VECTOR_COUNT);
        base_data_codes.resize((long long) NUM_BASE_VECTOR_EVQA);
        center_data.resize((long long) NUM_CLUSTER * VECTOR_DIM);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * VECTOR_DIM);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        load_from_evqa(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, EVQA_TEST_NUMBER, qrels);
        index_file = "../../example_index/evqaIndex" + std::to_string(NUM_GRAPH_CLUSTER_EVQA) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        save_result_file = "../../example_index/evqa_results_" + std::to_string(NUM_GRAPH_CLUSTER_EVQA) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
    } else if (dataset == 4) {
        NUM_BASE_SETS = NUM_BASE_SETS_SCIDOCS;
        NUM_QUERY_SETS = NUM_QUERT_SCIDOCS;
        NUM_CLUSTER = NUM_CLUSTER_SCIDOCS;
        NUM_GRAPH_CLUSTER = NUM_GRAPH_CLUSTER_SCIDOCS;
        dataset_path = dataset_path + "scidocs/";
        VECTOR_DIM = infer_vector_dim(dataset_path + "docdata/encoding0_float16.npy", dataset_path + "qdata/filterd_query.npy");
        {
            const std::pair<int, int> query_layout = infer_query_layout_from_lengths(dataset_path + "qdata/filterd_query_len.npy");
            NUM_QUERY_SETS = query_layout.first;
            QUERY_VECTOR_COUNT = query_layout.second;
        }
        std::cout << "Using vector dimension: " << VECTOR_DIM << std::endl;
        std::cout << "Using query count/vector count: " << NUM_QUERY_SETS << " / " << QUERY_VECTOR_COUNT << std::endl;
        base_data.resize((long long) NUM_BASE_VECTOR_SCIDOCS * VECTOR_DIM);
        query_data.resize((long long) NUM_BASE_VECTOR_SCIDOCS / 10 * VECTOR_DIM); // temporary, resized below
        query_data.resize((long long) get_npy_element_count(dataset_path + "qdata/filterd_query.npy"));
        base_data_codes.resize((long long) NUM_BASE_VECTOR_SCIDOCS);
        center_data.resize((long long) NUM_CLUSTER * VECTOR_DIM);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * VECTOR_DIM);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        load_from_scidocs(base_data, base, query_data, query, base_data_codes, center_data, graph_center_data, cluster_set, SCIDOCS_TEST_NUMBER, qrels);
        index_file = "../../example_index/scidocsIndex" + std::to_string(NUM_GRAPH_CLUSTER_SCIDOCS) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        save_result_file = "../../example_index/scidocs_results_" + std::to_string(NUM_GRAPH_CLUSTER_SCIDOCS) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
    } else if (dataset == TEST_GENERIC) {
        dataset_path = generic_dataset_root_override;
        if (dataset_path.empty()) {
            throw std::runtime_error("GEM_DATASET_PATH is required when GEM_DATASET is generic/nq/fiqa.");
        }
        if (dataset_path.back() != '/') {
            dataset_path.push_back('/');
        }
        const std::string docdata_path = dataset_path + "docdata/";
        const std::string qembs_path = dataset_path + "qdata/qembs.npy";
        const std::string filtered_qembs_path = dataset_path + "qdata/filterd_query.npy";
        const std::string filtered_qlens_path = dataset_path + "qdata/filterd_query_len.npy";
        const std::string centroids_path = dataset_path + "cdata/centroids.npy";
        const std::string coarse_centroids_path = dataset_path + "cdata/coarse_centroids.npy";
        const int shard_count = infer_doc_shard_count(docdata_path);
        const bool has_dense_qembs = file_exists(qembs_path);
        if (has_dense_qembs) {
            VECTOR_DIM = infer_vector_dim(docdata_path + "encoding0_float16.npy", qembs_path);
            const std::pair<int, int> query_layout = infer_query_layout(qembs_path);
            NUM_QUERY_SETS = query_layout.first;
            QUERY_VECTOR_COUNT = query_layout.second;
        } else {
            if (!file_exists(filtered_qembs_path) || !file_exists(filtered_qlens_path)) {
                throw std::runtime_error("Generic dataset is missing qdata/qembs.npy and qdata/filterd_query.npy.");
            }
            VECTOR_DIM = infer_vector_dim(docdata_path + "encoding0_float16.npy", filtered_qembs_path);
            const std::pair<int, int> query_layout = infer_query_layout_from_lengths(filtered_qlens_path);
            NUM_QUERY_SETS = query_layout.first;
            QUERY_VECTOR_COUNT = query_layout.second;
        }
        NUM_CLUSTER = generic_num_cluster_override > 0
            ? generic_num_cluster_override
            : static_cast<int>(read_npy_shape(centroids_path)[0]);
        NUM_GRAPH_CLUSTER = generic_num_graph_cluster_override > 0
            ? generic_num_graph_cluster_override
            : static_cast<int>(read_npy_shape(coarse_centroids_path)[0]);
        const long long num_base_vectors = infer_total_vectors_from_shards(docdata_path, shard_count);
        NUM_BASE_SETS = infer_total_docs_from_shards(docdata_path, shard_count);
        index_file = index_root_override + "/" + generic_dataset_name + "Index" + std::to_string(NUM_GRAPH_CLUSTER) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "/";
        if (!result_dir_override.empty()) {
            save_result_file = result_dir_override + "/" + generic_dataset_name + "_results_" + std::to_string(NUM_GRAPH_CLUSTER) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
        } else {
            save_result_file = index_root_override + "/" + generic_dataset_name + "_results_" + std::to_string(NUM_GRAPH_CLUSTER) + "_all_" + std::to_string(M_index) + "_" + std::to_string(EF_index) + "_" + std::to_string(K) + ".txt";
        }
        std::cout << index_file << std::endl;
        std::cout << "Using generic dataset name: " << generic_dataset_name << std::endl;
        std::cout << "Using generic dataset root: " << dataset_path << std::endl;
        std::cout << "Using vector dimension: " << VECTOR_DIM << std::endl;
        std::cout << "Using query count/vector count: " << NUM_QUERY_SETS << " / " << QUERY_VECTOR_COUNT << std::endl;
        std::cout << "Using base set/vector count: " << NUM_BASE_SETS << " / " << num_base_vectors << std::endl;
        std::cout << "Using cluster count/coarse count: " << NUM_CLUSTER << " / " << NUM_GRAPH_CLUSTER << std::endl;
        if (msmarco_base_fp32) {
            base_data.resize(num_base_vectors * VECTOR_DIM);
        } else {
            base_data_half.resize(num_base_vectors * VECTOR_DIM);
        }
        query_data.resize((long long) get_npy_element_count(has_dense_qembs ? qembs_path : filtered_qembs_path));
        base_data_codes.resize(num_base_vectors);
        center_data.resize((long long) NUM_CLUSTER * VECTOR_DIM);
        graph_center_data.resize((long long) NUM_GRAPH_CLUSTER * VECTOR_DIM);
        cluster_set.resize(NUM_GRAPH_CLUSTER);
        std::cout << "Generic base storage: " << (msmarco_base_fp32 ? "fp32" : "fp16") << " in RAM" << std::endl;
        load_from_msmarco(base_data, base_data_half, base, query_data, query, base_data_codes, center_data, graph_center_data,
                          cluster_set, shard_count, qrels, msmarco_base_fp32);
    }
    temp_cluster_id.resize(NUM_GRAPH_CLUSTER);
    for (int i = 0; i < NUM_GRAPH_CLUSTER; i++) {
        temp_cluster_id[i] = i;
    }
    NUM_CLUSTER_CALC_RUNTIME = NUM_CLUSTER;
    std::cout << temp_cluster_id.size() << std::endl;
    std::vector<float> col_query_cluster_scores(NUM_CLUSTER * QUERY_VECTOR_COUNT);
    test_query_cluster_scores.resize(NUM_CLUSTER * QUERY_VECTOR_COUNT);
    col_query_cluster_scores.resize(NUM_CLUSTER * QUERY_VECTOR_COUNT);

    std::string effective_shortcut_mode = shortcut_mode;
    if (effective_shortcut_mode.empty()) {
        effective_shortcut_mode = shortcut_edge_file.empty() ? "none" : "legacy";
    }
    if (effective_shortcut_mode != "none" &&
        effective_shortcut_mode != "legacy" &&
        effective_shortcut_mode != "paper") {
        throw std::runtime_error("Unsupported GEM_SHORTCUT_MODE: " + effective_shortcut_mode);
    }
    if (shortcut_topf <= 0 || shortcut_ef_search <= 0) {
        throw std::runtime_error("Shortcut topf/ef_search must be positive.");
    }
    if (effective_shortcut_mode == "paper") {
        if (paper_shortcut_train_qembs.empty() || paper_shortcut_train_qrels.empty()) {
            throw std::runtime_error("Paper shortcut mode requires GEM_PAPER_SHORTCUT_TRAIN_QEMBS and GEM_PAPER_SHORTCUT_TRAIN_QRELS.");
        }
        load_pair_aligned_query_pairs(paper_shortcut_train_qembs, paper_shortcut_train_qrels,
                                      shortcut_query_data, shortcut_train_query, shortcut_pairs);
        std::cout << "Paper shortcut mode enabled with " << shortcut_pairs.size()
                  << " sampled training pairs." << std::endl;
    } else if (effective_shortcut_mode == "legacy") {
        std::cout << "Legacy shortcut edge-file mode enabled." << std::endl;
    } else {
        std::cout << "Shortcut injection disabled." << std::endl;
    }
    std::cout << "cluster_entry_mode=" << cluster_entry_mode
              << " cluster_entry_seed=" << cluster_entry_seed
              << " shortcut_mode=" << effective_shortcut_mode
              << " shortcut_topf=" << shortcut_topf
              << " shortcut_ef_search=" << shortcut_ef_search
              << " apply_repair_on_build=" << apply_repair_on_build
              << " apply_repair_on_load=" << apply_repair_on_load << std::endl;
    std::cout << "ef_list:";
    for (const int ef_value : eflist) {
        std::cout << " " << ef_value;
    }
    std::cout << std::endl;
    std::cout << "rerank_list:";
    for (const int rerank_value : reranklist) {
        std::cout << " " << rerank_value;
    }
    std::cout << std::endl;
    std::cout << "nprob_list:";
    for (const int nprob_value : nproblist) {
        if (nprob_value <= 0 || nprob_value > NUM_GRAPH_CLUSTER) {
            throw std::runtime_error("GEM_NPROB/GEM_NPROB_LIST values must be in [1, NUM_GRAPH_CLUSTER].");
        }
        std::cout << " " << nprob_value;
    }
    std::cout << std::endl;
    std::cout << "query_profile_enabled=" << (query_profile_enabled ? 1 : 0) << std::endl;
    std::cout << "query_limit=" << query_limit << std::endl;

    if (build_thread_count > 0) {
        omp_set_num_threads(build_thread_count);
        openblas_set_num_threads(build_thread_count);
        std::cout << "Build thread limit: " << build_thread_count << std::endl;
    }

    Solution solution;
    solution.cluster_entry_seed = cluster_entry_seed;
    solution.cluster_entry_mode = cluster_entry_mode;
    if (rebuild) {
        if (cluster_distance_block_rows <= 0) {
            throw std::runtime_error("GEM_CLUSTER_DISTANCE_BLOCK_ROWS must be positive.");
        }
        {
            std::vector<uint16_t> cluster_distance((long long) NUM_CLUSTER * NUM_CLUSTER);
            const int block_rows = std::min(cluster_distance_block_rows, NUM_CLUSTER);
            std::vector<float> cluster_distance_block((long long) block_rows * NUM_CLUSTER);
            const int total_blocks = (NUM_CLUSTER + block_rows - 1) / block_rows;
            std::cout << "Cluster distance storage: fp16 blocks of " << block_rows << " rows" << std::endl;
            for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
                const int row_start = block_idx * block_rows;
                const int rows = std::min(block_rows, NUM_CLUSTER - row_start);
                if (block_idx % 16 == 0 || block_idx + 1 == total_blocks) {
                    std::cout << "cluster distance block " << (block_idx + 1) << "/" << total_blocks << std::endl;
                }
                hnswlib::fast_dot_product_blas(rows, VECTOR_DIM, NUM_CLUSTER,
                                               center_data.data() + (long long) row_start * VECTOR_DIM,
                                               center_data.data(),
                                               cluster_distance_block.data());
#pragma omp parallel for schedule(static)
                for (int row = 0; row < rows; ++row) {
                    encode_float_array_to_half(cluster_distance_block.data() + (long long) row * NUM_CLUSTER,
                                               cluster_distance.data() + ((long long) row_start + row) * NUM_CLUSTER,
                                               NUM_CLUSTER);
                }
            }
            solution.build_fine_cluster(VECTOR_DIM, base, cluster_set, temp_cluster_id,
                                        reinterpret_cast<const float*>(cluster_distance.data()));
        }
        if (apply_repair_on_build) {
            std::cout << "Applying repair_fine_graph_structure before save." << std::endl;
            solution.repair_fine_graph_structure(cluster_set);
        }
        if (effective_shortcut_mode == "legacy" && !shortcut_edge_file.empty()) {
            std::vector<std::pair<int, int>> edge_pair;
            load_edge_pairs_from_file(shortcut_edge_file, edge_pair);
            solution.apply_shortcut_edges(edge_pair);
        } else if (effective_shortcut_mode == "paper") {
            std::cout << "Applying paper shortcut injection on the live graph." << std::endl;
            solution.apply_paper_shortcut_pairs(shortcut_train_query, shortcut_pairs,
                                                test_query_cluster_scores, col_query_cluster_scores,
                                                center_data, graph_center_data,
                                                shortcut_topf, shortcut_ef_search,
                                                static_cast<size_t>(M_index));
        }
        solution.save_fine_cluster(index_file);
        write_index_metadata(index_file, VECTOR_DIM, NUM_BASE_SETS, NUM_QUERY_SETS, NUM_CLUSTER, NUM_GRAPH_CLUSTER);
    } else {
        validate_index_metadata_or_throw(index_file, VECTOR_DIM, NUM_BASE_SETS, NUM_QUERY_SETS, NUM_CLUSTER, NUM_GRAPH_CLUSTER);
        solution.load_fine_cluster(index_file, VECTOR_DIM, base, cluster_set, temp_cluster_id);
        if (apply_repair_on_load) {
            std::cout << "Applying repair_fine_graph_structure after load." << std::endl;
            solution.repair_fine_graph_structure(cluster_set);
        }
        if (effective_shortcut_mode == "paper") {
            std::cout << "Applying paper shortcut injection on the loaded graph." << std::endl;
            solution.apply_paper_shortcut_pairs(shortcut_train_query, shortcut_pairs,
                                                test_query_cluster_scores, col_query_cluster_scores,
                                                center_data, graph_center_data,
                                                shortcut_topf, shortcut_ef_search,
                                                static_cast<size_t>(M_index));
        }
    }

    if (skip_search) {
        std::cout << "GEM_SKIP_SEARCH=1; index build/load finished and query search is skipped." << std::endl;
        return 0;
    }

    if (search_thread_count <= 0) {
        throw std::runtime_error("GEM_SEARCH_THREADS must be positive.");
    }
    if (query_limit < 0) {
        throw std::runtime_error("GEM_QUERY_LIMIT must be non-negative.");
    }
    const int active_num_query_sets = query_limit > 0 ? std::min(query_limit, NUM_QUERY_SETS) : NUM_QUERY_SETS;
    std::cout << "Active query count: " << active_num_query_sets
              << " / " << NUM_QUERY_SETS << std::endl;
    omp_set_num_threads(search_thread_count);
    openblas_set_num_threads(search_thread_count);
    std::cout << "Search thread limit: " << search_thread_count << std::endl;

    for (int nprob_value: nproblist) {
        NPROB = nprob_value;
        std::cout << "Query cluster filter t/NPROB: " << NPROB << std::endl;
        for (int r: reranklist) {
            rerankK = r;
            for (int tmpef: eflist) {
            std::ofstream result_file;
            std::string run_result_file;
            if (save_result) {
                run_result_file = save_result_file;
                const std::string suffix = "_t" + std::to_string(NPROB) + "_rerank" + std::to_string(rerankK) + "_ef" + std::to_string(tmpef) + ".tsv";
                const size_t ext_pos = run_result_file.rfind(".txt");
                if (ext_pos != std::string::npos) {
                    run_result_file.replace(ext_pos, 4, suffix);
                } else {
                    run_result_file += suffix;
                }
                result_file.open(run_result_file, std::ios::out | std::ios::trunc);
                if (!result_file.is_open()) {
                    throw std::runtime_error("Failed to open result file: " + run_result_file);
                }
                std::cout << "Writing TSV results to: " << run_result_file << std::endl;
            }
            std::ofstream query_profile_file;
            if (query_profile_enabled) {
                const std::string query_profile_path = make_query_profile_path(run_result_file);
                query_profile_file.open(query_profile_path, std::ios::out | std::ios::trunc);
                if (!query_profile_file.is_open()) {
                    throw std::runtime_error("Failed to open query profile file: " + query_profile_path);
                }
                write_query_profile_header(query_profile_file);
                std::cout << "Writing query profiles to: " << query_profile_path << std::endl;
            }
            double total_dataset_hnsw_recall = 0.0;
            double total_query_time = 0.0;
            double total_dataset_hnsw_mrr = 0.0;
            double total_dataset_hitrate = 0.0;
            const double search_ram_start_mib = get_current_ram_mebibytes();
            double search_peak_ram_mib = search_ram_start_mib;

            std::cout<<"Processing Queries HNSW"<<std::endl;
        
            for (int i = 0; i < active_num_query_sets; ++i) {
                std::vector<std::pair<int, float>> solution_indices;
                QuerySearchProfile query_profile;
                double query_time = solution.search_with_fine_cluster(query[i], test_query_cluster_scores, col_query_cluster_scores,
                                                                      center_data, graph_center_data, K, tmpef,
                                                                      solution_indices,
                                                                      query_profile_enabled ? &query_profile : nullptr);
                total_query_time += query_time;
                const double current_search_ram_mib = get_current_ram_mebibytes();
                if (current_search_ram_mib >= 0.0) {
                    search_peak_ram_mib = std::max(search_peak_ram_mib, current_search_ram_mib);
                }
                bool query_has_qrels = i < qrels.size() && !qrels[i].empty();
                double dataset_hnsw_recall = 0.0;
                double dataset_hnsw_mrr = 0.0;
                double dataset_hitrate = 0.0;
                int first_hit_rank = -1;
                size_t topk_relevant_hits = 0;
                if (query_has_qrels) {
                    dataset_hnsw_recall = calculate_recall_for_datasetlabel(solution_indices, qrels[i]);
                    dataset_hnsw_mrr = calculate_mrr_for_datasetlabel(solution_indices, qrels[i]);
                    dataset_hitrate = calculate_hitrate_for_datasetlabel(solution_indices, qrels[i]);
                    first_hit_rank = calculate_first_hit_rank_for_datasetlabel(solution_indices, qrels[i]);
                    topk_relevant_hits = calculate_hit_count_for_datasetlabel(solution_indices, qrels[i]);
                }
                if (query_profile_enabled) {
                    write_query_profile_row(query_profile_file, i, NPROB, tmpef, rerankK, query_profile,
                                            query_has_qrels,
                                            query_has_qrels ? qrels[i].size() : 0,
                                            topk_relevant_hits,
                                            first_hit_rank,
                                            dataset_hnsw_recall,
                                            dataset_hnsw_mrr,
                                            dataset_hitrate);
                }
                if (save_result) {
                    for (size_t rank = 0; rank < solution_indices.size(); ++rank) {
                        result_file << i << '\t'
                                    << solution_indices[rank].first << '\t'
                                    << (-solution_indices[rank].second) << '\t'
                                    << (rank + 1) << '\n';
                    }
                }
                if (query_has_qrels) {
                    total_dataset_hnsw_recall += dataset_hnsw_recall;
                    total_dataset_hnsw_mrr += dataset_hnsw_mrr;
                    total_dataset_hitrate += dataset_hitrate;
                    std::cout << "Recall for query set " << i << ": " << dataset_hnsw_recall << " " << dataset_hnsw_mrr << " | " << query_time << std::endl;
                } else {
                    std::cout << "Query set " << i << " processed | " << query_time << std::endl;
                }
            }
            std::cout << "t/NPROB: " << NPROB << " rerankK: " << rerankK << " ef: " << tmpef << std::endl;
            bool has_qrels = false;
            for (int i = 0; i < active_num_query_sets && i < static_cast<int>(qrels.size()); ++i) {
                if (!qrels[i].empty()) {
                    has_qrels = true;
                    break;
                }
            }
            if (has_qrels) {
                std::cout << "Average our method recall v.s. dataset label: " << total_dataset_hnsw_recall / active_num_query_sets << std::endl;
                std::cout << "Average our method mrr v.s. dataset label: " << total_dataset_hnsw_mrr/ active_num_query_sets << std::endl;
                std::cout << "Average our method hitrate v.s. dataset label: " << total_dataset_hitrate/ active_num_query_sets << std::endl;
            } else {
                std::cout << "No qrels loaded; skipping metric aggregation." << std::endl;
            }
            const double avg_query_time = total_query_time / active_num_query_sets;
            const double qps = avg_query_time > 0.0 ? (1.0 / avg_query_time) : 0.0;
            const double peak_ram_mib = get_peak_ram_mebibytes();
            std::cout << "Average query time: " << avg_query_time << " seconds" << std::endl;
            std::cout << "QPS: " << qps << std::endl;
            if (peak_ram_mib >= 0.0) {
                std::cout << "Peak RAM: " << peak_ram_mib << " MiB" << std::endl;
            } else {
                std::cout << "Peak RAM: unavailable" << std::endl;
            }
            if (search_peak_ram_mib >= 0.0) {
                std::cout << "Peak RAM During Search: " << search_peak_ram_mib << " MiB" << std::endl;
                if (search_ram_start_mib >= 0.0) {
                    std::cout << "Search RAM Delta: " << (search_peak_ram_mib - search_ram_start_mib) << " MiB" << std::endl;
                }
            } else {
                std::cout << "Peak RAM During Search: unavailable" << std::endl;
            }
            if (save_result) {
                result_file.close();
                if (query_profile_enabled) {
                    query_profile_file.close();
                }
                const std::string meta_file = run_result_file + ".meta.json";
                std::ofstream meta_stream(meta_file, std::ios::out | std::ios::trunc);
                if (!meta_stream.is_open()) {
                    throw std::runtime_error("Failed to open metadata file: " + meta_file);
                }
                meta_stream << "{\n"
                            << "  \"run\": \"" << run_result_file << "\",\n"
                            << "  \"nprob\": " << NPROB << ",\n"
                            << "  \"rerankK\": " << rerankK << ",\n"
                            << "  \"ef\": " << tmpef << ",\n"
                            << "  \"num_queries\": " << active_num_query_sets << ",\n"
                            << "  \"dataset_num_queries\": " << NUM_QUERY_SETS << ",\n"
                            << "  \"query_limit\": " << query_limit << ",\n"
                            << "  \"avg_query_time_seconds\": " << avg_query_time << ",\n"
                            << "  \"qps\": " << qps << ",\n"
                            << "  \"peak_ram_mib\": " << peak_ram_mib << ",\n"
                            << "  \"search_ram_start_mib\": " << search_ram_start_mib << ",\n"
                            << "  \"search_peak_ram_mib\": " << search_peak_ram_mib << ",\n"
                            << "  \"search_ram_delta_mib\": "
                            << ((search_peak_ram_mib >= 0.0 && search_ram_start_mib >= 0.0)
                                    ? (search_peak_ram_mib - search_ram_start_mib)
                                    : -1.0)
                            << "\n"
                            << "}\n";
                meta_stream.close();
            }
        }
    }
    }
    return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal exception: " << ex.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Fatal exception: unknown" << std::endl;
        return 1;
    }
}
