#pragma once
#include <fstream>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cstdint>

#include "MultiVectorDoc.hpp"   // 提供 vdata_t / _data_doc / MultiVectorDoc
#include "utils.hpp"            // 提供 VidType / DidType（与你的 FileIO.hpp 保持一致）

namespace fivecs_io {

// 与 FileIO.hpp::ReadHeader 对称：头部依次写 nvecs, ndocs, dim（均以 VidType 宽度落盘）
inline void write_header(std::ofstream& ofs, VidType nvecs, VidType ndocs, VidType dim) {
    ofs.write(reinterpret_cast<const char*>(&nvecs), sizeof(VidType));
    ofs.write(reinterpret_cast<const char*>(&ndocs), sizeof(VidType));
    ofs.write(reinterpret_cast<const char*>(&dim),   sizeof(VidType));
    if (!ofs) throw std::runtime_error("fivecs header write failed");
}

// 计算总向量数并做基本一致性检查（所有 doc 的 dim 必须一致）
inline void _check_and_count(const std::vector<MultiVectorDoc*>& docs,
                             size_t& dim0, uint64_t& total_vecs) {
    if (docs.empty()) throw std::invalid_argument("no docs to save");
    dim0 = docs.front()->dim;
    if (dim0 == 0) throw std::invalid_argument("dim=0 is invalid");
    total_vecs = 0;
    for (auto* d : docs) {
        if (!d) throw std::invalid_argument("null MultiVectorDoc*");
        if (d->dim != dim0) throw std::invalid_argument("inconsistent doc dim");
        total_vecs += d->doclen;
    }
    if (total_vecs > static_cast<uint64_t>(std::numeric_limits<VidType>::max()))
        throw std::overflow_error("nvecs exceeds VidType range");
    if (docs.size() > static_cast<size_t>(std::numeric_limits<VidType>::max()))
        throw std::overflow_error("ndocs exceeds VidType range");
    if (dim0 > static_cast<size_t>(std::numeric_limits<VidType>::max()))
        throw std::overflow_error("dim exceeds VidType range");
}

inline MultiVectorDoc* _alloc_mvdoc(size_t doclen, size_t dim) {
    const size_t bytes = _size_doc(doclen, dim);
    void* mem = ::operator new(bytes);                    // 抛出 std::bad_alloc
    auto* p = new (mem) MultiVectorDoc();                 // 仅构造头部字段
    p->doclen = doclen;
    p->dim = dim;
    return p;
}

inline void _free_mvdoc(MultiVectorDoc* p) {
    if (!p) return;
    p->~MultiVectorDoc();
    ::operator delete(p);
}

// // 块式两遍扫描：每遍都用大块顺序读 + 内存内解析，避免逐记录 read/seek。
// inline std::vector<MultiVectorDoc*>
// ReadMultiVectorDocFromFivecs(const std::string& file_name,
//                                             size_t chunk_bytes = 64ull << 20 /* 64MB */) {
//     std::ifstream ifs(file_name, std::ios::binary);
//     if (!ifs.is_open()) throw std::runtime_error("cannot open: " + file_name);

//     VidType nvecs_v = 0, ndocs_v = 0, dim_v = 0;
//     ifs.read(reinterpret_cast<char*>(&nvecs_v), sizeof(VidType));
//     ifs.read(reinterpret_cast<char*>(&ndocs_v), sizeof(VidType));
//     ifs.read(reinterpret_cast<char*>(&dim_v),   sizeof(VidType));
//     if (!ifs) throw std::runtime_error("failed to read header");

//     const uint64_t nvecs = static_cast<uint64_t>(nvecs_v);
//     const size_t   ndocs = static_cast<size_t>(ndocs_v);
//     const size_t   dim   = static_cast<size_t>(dim_v);
//     if (dim == 0) return {};

//     // 每条记录的字节数：vid + did + dim * vdata_t
//     const size_t rec_bytes = sizeof(VidType) + sizeof(VidType) + sizeof(vdata_t) * dim;
//     // 对齐 chunk 大小，至少能容纳一条
//     if (chunk_bytes < rec_bytes) chunk_bytes = rec_bytes;
//     chunk_bytes -= (chunk_bytes % rec_bytes);

//     std::vector<size_t> counts(ndocs, 0);

//     // -------- 第一遍：计数 --------
//     {
//         std::vector<unsigned char> buf(chunk_bytes);
//         // 记录体起点
//         std::streamoff body_off = static_cast<std::streamoff>(sizeof(VidType) * 3);
//         ifs.seekg(body_off, std::ios::beg);
//         if (!ifs) throw std::runtime_error("seek to body failed");

//         uint64_t seen = 0;
//         while (seen < nvecs) {
//             uint64_t remain_recs = nvecs - seen;
//             size_t want_bytes = static_cast<size_t>(std::min<uint64_t>(remain_recs, chunk_bytes / rec_bytes)) * rec_bytes;
//             ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(want_bytes));
//             if (!ifs) throw std::runtime_error("read body chunk failed (pass1)");
//             const unsigned char* p = buf.data();
//             const unsigned char* e = buf.data() + want_bytes;
//             while (p < e) {
//                 // 解析 did（跳过 vid）
//                 p += sizeof(VidType);
//                 VidType did_v;
//                 std::memcpy(&did_v, p, sizeof(VidType));
//                 p += sizeof(VidType);
//                 const size_t did = static_cast<size_t>(did_v);
//                 if (did >= ndocs) throw std::runtime_error("did out of range (pass1)");
//                 ++counts[did];
//                 // 跳过 payload
//                 p += sizeof(vdata_t) * dim;
//             }
//             seen += static_cast<uint64_t>(want_bytes / rec_bytes);
//         }
//     }

//     // 分配输出
//     std::vector<MultiVectorDoc*> docs(ndocs, nullptr);
//     try {
//         for (size_t d = 0; d < ndocs; ++d) docs[d] = _alloc_mvdoc(counts[d], dim);
//     } catch (...) {
//         for (auto* p : docs) _free_mvdoc(p);
//         throw;
//     }
//     std::vector<size_t> cursor(ndocs, 0);

//     // -------- 第二遍：填充 --------
//     ifs.clear();
//     ifs.seekg(static_cast<std::streamoff>(sizeof(VidType) * 3), std::ios::beg);
//     if (!ifs) { for (auto* p : docs) _free_mvdoc(p); throw std::runtime_error("seek to body failed (pass2)"); }

//     {
//         std::vector<unsigned char> buf(chunk_bytes);
//         uint64_t seen = 0;
//         while (seen < nvecs) {
//             uint64_t remain_recs = nvecs - seen;
//             size_t want_bytes = static_cast<size_t>(std::min<uint64_t>(remain_recs, chunk_bytes / rec_bytes)) * rec_bytes;
//             ifs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(want_bytes));
//             if (!ifs) { for (auto* p : docs) _free_mvdoc(p); throw std::runtime_error("read body chunk failed (pass2)"); }

//             const unsigned char* p = buf.data();
//             const unsigned char* e = buf.data() + want_bytes;
//             while (p < e) {
//                 // 取 did
//                 p += sizeof(VidType);
//                 VidType did_v;
//                 std::memcpy(&did_v, p, sizeof(VidType));
//                 p += sizeof(VidType);
//                 const size_t did = static_cast<size_t>(did_v);

//                 // 拷贝 payload 到目标 doc 的当前位置
//                 vdata_t* base = _data_doc(docs[did]);
//                 vdata_t* dest = base + cursor[did] * dim;
//                 std::memcpy(dest, p, sizeof(vdata_t) * dim);
//                 p += sizeof(vdata_t) * dim;
//                 ++cursor[did];
//             }
//             seen += static_cast<uint64_t>(want_bytes / rec_bytes);
//         }
//     }

// #ifndef NDEBUG
//     // 轻量一致性检查
//     for (size_t d = 0; d < ndocs; ++d) {
//         if (!(cursor[d] == counts[d] && docs[d]->doclen == counts[d] && docs[d]->dim == dim)) {
//             for (auto* p : docs) _free_mvdoc(p);
//             throw std::runtime_error("doc consistency check failed");
//         }
//     }
// #endif
//     return docs;
// }

inline std::vector<MultiVectorDoc*>
ReadMultiVectorDocFromFivecs(const std::string& file_name) {
    std::ifstream ifs(file_name, std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("cannot open file: " + file_name);

    // 头部：与你的 fivecs 约定对齐，三者都以 VidType 宽度落盘
    VidType nvecs_v = 0, ndocs_v = 0, dim_v = 0;
    ifs.read(reinterpret_cast<char*>(&nvecs_v), sizeof(VidType));
    ifs.read(reinterpret_cast<char*>(&ndocs_v), sizeof(VidType));
    ifs.read(reinterpret_cast<char*>(&dim_v),   sizeof(VidType));
    if (!ifs) throw std::runtime_error("failed to read fivecs header");

    const uint64_t nvecs = static_cast<uint64_t>(nvecs_v);
    const size_t   ndocs = static_cast<size_t>(ndocs_v);
    const size_t   dim   = static_cast<size_t>(dim_v);
    if (dim == 0) throw std::invalid_argument("fivecs dim = 0");
    if (ndocs == 0 && nvecs != 0) throw std::invalid_argument("ndocs=0 but nvecs>0");

    // 第一遍：只统计每个 did 的 doclen
    std::vector<size_t> counts(ndocs, 0);
    for (uint64_t i = 0; i < nvecs; ++i) {
        VidType vid_tmp = 0, did_tmp = 0;
        ifs.read(reinterpret_cast<char*>(&vid_tmp), sizeof(VidType));
        ifs.read(reinterpret_cast<char*>(&did_tmp), sizeof(VidType));
        if (!ifs) throw std::runtime_error("failed to read (vid,did) at record " + std::to_string(i));
        const size_t did = static_cast<size_t>(did_tmp);
        if (did >= ndocs) throw std::runtime_error("did out of range at record " + std::to_string(i));
        // 跳过 dim 个标量
        ifs.seekg(static_cast<std::streamoff>(sizeof(vdata_t) * dim), std::ios::cur);
        if (!ifs) throw std::runtime_error("failed to skip payload at record " + std::to_string(i));
        ++counts[did];
    }

    // 为每个文档一次性分配连续内存
    std::vector<MultiVectorDoc*> docs; docs.resize(ndocs, nullptr);
    try {
        for (size_t d = 0; d < ndocs; ++d)
            docs[d] = _alloc_mvdoc(counts[d], dim);
    } catch (...) {
        for (auto* p : docs) _free_mvdoc(p);
        throw;
    }
    // 写入位置指针（单位：向量条目）
    std::vector<size_t> cursor(ndocs, 0);

    // 回到文件体起点，进行第二遍填充
    const std::streamoff body_offset =
        static_cast<std::streamoff>(sizeof(VidType) * 3); // 头部长度
    ifs.clear();
    ifs.seekg(body_offset, std::ios::beg);
    if (!ifs) {
        for (auto* p : docs) _free_mvdoc(p);
        throw std::runtime_error("seek back to body failed");
    }

    std::vector<vdata_t> tmp(dim); // 小缓冲，避免逐标量 read 造成调用开销
    for (uint64_t i = 0; i < nvecs; ++i) {
        VidType vid_tmp = 0, did_tmp = 0;
        ifs.read(reinterpret_cast<char*>(&vid_tmp), sizeof(VidType));
        ifs.read(reinterpret_cast<char*>(&did_tmp), sizeof(VidType));
        if (!ifs) { for (auto* p : docs) _free_mvdoc(p); throw std::runtime_error("failed to reread (vid,did) at record " + std::to_string(i)); }

        const size_t did = static_cast<size_t>(did_tmp);
        if (did >= ndocs) { for (auto* p : docs) _free_mvdoc(p); throw std::runtime_error("did out of range on second pass"); }
        if (cursor[did] >= counts[did]) { for (auto* p : docs) _free_mvdoc(p); throw std::runtime_error("overflow for doc " + std::to_string(did)); }

        vdata_t* base = _data_doc(docs[did]);
        vdata_t* dest = base + cursor[did] * dim;

        ifs.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(sizeof(vdata_t) * dim));
        if (!ifs) { for (auto* p : docs) _free_mvdoc(p); throw std::runtime_error("failed to read payload at record " + std::to_string(i)); }
        std::memcpy(dest, tmp.data(), sizeof(vdata_t) * dim);

        ++cursor[did];
    }

// #ifndef NDEBUG
//     // 轻量一致性检查
//     for (size_t d = 0; d < ndocs; ++d) {
//         if (!(cursor[d] == counts[d] && docs[d]->doclen == counts[d] && docs[d]->dim == dim)) {
//             for (auto* p : docs) _free_mvdoc(p);
//             throw std::runtime_error("doc consistency check failed");
//         }
//     }
// #endif
    return docs;
}



// 核心：将一批 MultiVectorDoc 直接写为 fivecs（二进制）
inline void SaveMultiVectorDocsToFivecs(const std::string& file_name,
                                        const std::vector<MultiVectorDoc*>& docs)
{
    size_t dim0 = 0; uint64_t total_vecs = 0;
    _check_and_count(docs, dim0, total_vecs);

    std::ofstream ofs(file_name, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open())
        throw std::runtime_error("cannot open file: " + file_name);

    // 头部：三元组（与 FileIO.hpp::ReadHeader 对应，按 VidType 写）
    write_header(ofs,
                 static_cast<VidType>(total_vecs),
                 static_cast<VidType>(docs.size()),
                 static_cast<VidType>(dim0));

    // 主体：逐向量写 (vid, did, dim 个标量)
    VidType vid = 0;
    for (VidType did = 0; did < static_cast<VidType>(docs.size()); ++did) {
        const MultiVectorDoc* d = docs[did];
        const vdata_t* base = _data_doc(d);              // 连续区首地址
        const size_t dim = d->dim;
        for (size_t j = 0; j < d->doclen; ++j, ++vid) {
            const vdata_t* vecj = base + j * dim;
            // 先写 vid，再写 did（两者在 FileIO.hpp 中均按 VidType 读取）
            ofs.write(reinterpret_cast<const char*>(&vid), sizeof(VidType));
            ofs.write(reinterpret_cast<const char*>(&did), sizeof(VidType));
            ofs.write(reinterpret_cast<const char*>(vecj), sizeof(vdata_t) * dim);
            if (!ofs) throw std::runtime_error("fivecs body write failed");
        }
    }
    ofs.flush();
    if (!ofs) throw std::runtime_error("flush failed");
}

// // 便捷重载：对象容器
// inline void SaveMultiVectorDocsToFivecs(const std::string& file_name,
//                                         const std::vector<MultiVectorDoc>& docs_obj)
// {
//     std::vector<MultiVectorDoc*> ptrs; ptrs.reserve(docs_obj.size());
//     for (auto& d : docs_obj) ptrs.push_back(const_cast<MultiVectorDoc*>(&d));
//     SaveMultiVectorDocsToFivecs(file_name, ptrs);
// }

// 单文档版本（ndocs=1）
inline void SaveSingleDocToFivecs(const std::string& file_name,
                                  const MultiVectorDoc& doc)
{
    std::vector<MultiVectorDoc*> one{const_cast<MultiVectorDoc*>(&doc)};
    SaveMultiVectorDocsToFivecs(file_name, one);
}

} // namespace fivecs_io
