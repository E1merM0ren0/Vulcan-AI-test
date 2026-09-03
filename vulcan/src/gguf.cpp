// gguf.cpp — GGUF v3 binary writer.
//
// Layout (all little-endian):
//   magic "GGUF" | version(3) | tensor_count | metadata_kv_count
//   general.architecture string kv
//   ... user meta kvs (string=8, uint32=6, float32=7), each prefixed with size
//   tensor_info[n]: name | ndim | dims[] | type | data_offset
//   padding to 32 bytes, then tensor data made 32-byte aligned
// Each tensor's data is preceded by padding so its offset is a multiple of 32.

#include "gguf.h"
#include <cstring>
#include <fstream>
#include <sstream>

namespace gguf {

static void writeU32(std::ofstream& f, uint32_t v) {
    f.write((const char*)&v, 4);
}
static void writeU64(std::ofstream& f, uint64_t v) {
    f.write((const char*)&v, 8);
}
static void writeF32(std::ofstream& f, float v) {
    f.write((const char*)&v, 4);
}
static void writeString(std::ofstream& f, const std::string& s) {
    writeU64(f, (uint64_t)s.size());
    f.write(s.data(), (std::streamsize)s.size());
}

static uint64_t alignUp(uint64_t v, uint64_t align) {
    return (v + align - 1) & ~(align - 1);
}

bool writeGGUF(const std::string& path,
               const std::string& arch,
               const GGUFMeta& meta,
               const std::vector<TensorInfo>& tensors,
               std::string* err) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err) *err = "cannot open " + path + " for writing";
        return false;
    }

    uint64_t kvCount = 1; // general.architecture
    kvCount += meta.strings.size() + meta.u32s.size() + meta.f32s.size()
             + meta.strArrays.size() + meta.i32Arrays.size();

    writeU32(f, 0x46554747);
    writeU32(f, 3);
    writeU64(f, (uint64_t)tensors.size());
    writeU64(f, kvCount);

    writeString(f, "general.architecture");
    writeU32(f, 8); // GGML_KV_META_TYPE_STRING
    writeString(f, arch);

    for (const auto& kv : meta.strings) {
        writeString(f, kv.first);
        writeU32(f, 8);
        writeString(f, kv.second);
    }
    for (const auto& kv : meta.u32s) {
        writeString(f, kv.first);
        writeU32(f, 4); // GGUF_TYPE_UINT32
        writeU32(f, kv.second);
    }
    for (const auto& kv : meta.f32s) {
        writeString(f, kv.first);
        writeU32(f, 6); // GGUF_TYPE_FLOAT32
        writeF32(f, kv.second);
    }
    for (const auto& kv : meta.strArrays) {
        writeString(f, kv.first);
        writeU32(f, 9); // GGUF_TYPE_ARRAY
        writeU32(f, 8); // element type: GGUF_TYPE_STRING
        writeU64(f, (uint64_t)kv.second.size());
        for (const auto& v : kv.second) writeString(f, v);
    }
    for (const auto& kv : meta.i32Arrays) {
        writeString(f, kv.first);
        writeU32(f, 9); // GGUF_TYPE_ARRAY
        writeU32(f, 5); // element type: GGUF_TYPE_INT32
        writeU64(f, (uint64_t)kv.second.size());
        for (int32_t v : kv.second) writeU32(f, (uint32_t)v);
    }

    std::vector<uint64_t> offsets(tensors.size());
    uint64_t dataOffset = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        const TensorInfo& t = tensors[i];
        writeString(f, t.name);
        writeU32(f, (uint32_t)t.dims.size());
        for (uint32_t d : t.dims) writeU64(f, (uint64_t)d);
        writeU32(f, t.type);
        offsets[i] = dataOffset;
        writeU64(f, dataOffset);
        uint64_t byteSize = 4;
        for (uint32_t d : t.dims) byteSize *= (uint64_t)d;
        dataOffset += alignUp(byteSize, 32);
    }

    std::vector<char> padding(32, 0);

    // The data section must begin on a 32-byte boundary, so pad out whatever
    // the header (magic + counts + KV + tensor-info) ended on.
    uint64_t headerPos = (uint64_t)f.tellp();
    uint64_t padStart = alignUp(headerPos, 32) - headerPos;
    if (padStart > 0) {
        f.write(padding.data(), (std::streamsize)padStart);
    }

    for (size_t i = 0; i < tensors.size(); i++) {
        const TensorInfo& t = tensors[i];
        uint64_t byteSize = 4;
        for (uint32_t d : t.dims) byteSize *= (uint64_t)d;
        f.write((const char*)t.data.data(), (std::streamsize)byteSize);
        uint64_t rem = alignUp(byteSize, 32) - byteSize;
        if (rem > 0) f.write(padding.data(), (std::streamsize)rem);
    }

    f.close();
    if (err) err->clear();
    return true;
}

}
