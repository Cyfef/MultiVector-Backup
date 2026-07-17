#include <tuple>
#include <queue>
#include <types.hpp>
#include <cstdint>
#include <faiss/IndexFlat.h>
#include <algorithm>
#include <fstream>
#include <types.hpp>
using namespace std;

class Compare
{
public:
    bool operator()(const tuple<numDocsType, valType> &t1, const tuple<numDocsType, valType> &t2)
    {
        // return get<1>(t1) < get<1>(t2); // OLD
        return get<1>(t1) > get<1>(t2);
    }
};

class CompareInteger
{
public:
    bool operator()(const tuple<numDocsType, size_t> &t1, const tuple<numDocsType, size_t> &t2)
    {
        // return get<1>(t1) < get<1>(t2); // OLD
        return get<1>(t1) > get<1>(t2);
    }
};

// typedef priority_queue<tuple<numDocsType, valType>, vector<tuple<numDocsType, valType>>, Compare> heap_struct;

template <typename T>
vector<size_t> argsort(const vector<T> &array, const size_t start, const size_t offset)
{
    vector<size_t> indices(offset);
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(),
         [&array, start](int left, int right) -> bool
         {
             // sort indices according to corresponding array element
             return array[start + left] > array[start + right]; // now is descending order
         });

    return indices;
}

template <typename T>
inline vector<size_t> argpartition(const vector<T> &array, const size_t start, const size_t offset, const size_t k, vector<size_t> &unsorted_indexes)
{
    vector<size_t> out(k);
    iota(unsorted_indexes.begin(), unsorted_indexes.end(), 0);
    nth_element(unsorted_indexes.begin(), unsorted_indexes.begin() + k, unsorted_indexes.end(),
                [&array, start](int left, int right) -> bool
                {
                    // sort indices according to corresponding array element
                    return array[start + left] > array[start + right]; // now is descending order
                });

    copy(unsorted_indexes.begin(), unsorted_indexes.begin() + k, out.begin());
    return unsorted_indexes;
}

vector<uint32_t> load_qids(string path)
{
    vector<uint32_t> qid_map;
    ifstream qid_map_file(path);
    string line;
    while (getline(qid_map_file, line))
        qid_map.push_back(stoi(line));
    qid_map_file.close();
    qid_map.shrink_to_fit();
    return qid_map;
}


template <typename T>
inline vector<size_t> argpartition(const vector<T> &array, const size_t k, vector<size_t> &unsorted_indexes)
{
    vector<size_t> out(k);
    iota(unsorted_indexes.begin(), unsorted_indexes.end(), 0);
    nth_element(unsorted_indexes.begin(), unsorted_indexes.begin() + k, unsorted_indexes.end(),
                [&array](size_t left, size_t right) -> bool
                {
                    // sort indices according to corresponding array element
                    return array[left] > array[right]; // now is descending order
                });

    copy(unsorted_indexes.begin(), unsorted_indexes.begin() + k, out.begin());
    return out;
}


  inline void set_bit_64(const size_t doc_id, vector<uint64_t>& bitvectors_centroids)
    {
        size_t slot = doc_id / 64;
        size_t offset = doc_id % 64;

        bitvectors_centroids[slot] |= (uint64_t)1 << offset;
    }

    inline uint64_t check_bit_64(const size_t doc_id, const vector<uint64_t>& bitvectors_centroids)
    {
        size_t slot = doc_id / 64;
        size_t offset = doc_id % 64;

        return (bitvectors_centroids[slot] >> offset) & (uint64_t)1;
    }

    inline size_t popcount_u64(const uint64_t value)
    {
        return static_cast<size_t>(__builtin_popcountll(value));
    }

    void reset_bitvectors(vector<uint64_t> &bitvectors)
    {
        fill(bitvectors.begin(), bitvectors.end(), uint64_t{0});
    }

    inline void set_bit(
        const size_t q_term_id,
        const size_t centroid_id,
        const size_t words_per_centroid,
        vector<uint64_t> &bitvectors)
    {
        const size_t word_idx = q_term_id / 64;
        const size_t bit_idx = q_term_id % 64;
        bitvectors[centroid_id * words_per_centroid + word_idx] |= uint64_t{1} << bit_idx;
    }

    void init_bitvectors(
        const size_t n_centroids,
        const size_t words_per_centroid,
        vector<uint64_t> &bitvectors)
    {
        bitvectors.assign(n_centroids * words_per_centroid, uint64_t{0});
    }

    void assign_bitvectors(
        const vector<size_t> &sorted_indexes,
        const size_t topt,
        const size_t query_term_index,
        const size_t words_per_centroid,
        vector<uint64_t> &bitvectors)
    {
        for (size_t i = 0; i < topt; i++)
        {
            set_bit(query_term_index, sorted_indexes[i], words_per_centroid, bitvectors);
        }
    }

    void assign_bitvectors(
        size_t *sorted_indexes,
        const size_t topt,
        const size_t query_term_index,
        const size_t words_per_centroid,
        vector<uint64_t> &bitvectors)
    {
        for (size_t i = 0; i < topt; i++)
        {
            set_bit(query_term_index, sorted_indexes[i], words_per_centroid, bitvectors);
        }
    }
