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
    kvCount += meta.strings.size() + meta.u32s.size() + meta.f32s.size();

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
        writeU32(f, 6); // GGML_KV_META_TYPE_UINT32
        writeU32(f, kv.second);
    }
    for (const auto& kv : meta.f32s) {
        writeString(f, kv.first);
        writeU32(f, 7); // GGML_KV_META_TYPE_FLOAT32
        writeF32(f, kv.second);
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

    uint64_t pad = 0;
    std::vector<char> padding(32, 0);
    for (size_t i = 0; i < tensors.size(); i++) {
        const TensorInfo& t = tensors[i];
        uint64_t byteSize = 4;
        for (uint32_t d : t.dims) byteSize *= (uint64_t)d;
        if (offsets[i] % 32 != 0) {
            f.write(padding.data(), (std::streamsize)(32 - (long)(offsets[i] % 32)));
        }
        f.write((const char*)t.data.data(), (std::streamsize)byteSize);
        uint64_t rem = alignUp(byteSize, 32) - byteSize;
        if (rem > 0) f.write(padding.data(), (std::streamsize)rem);
    }

    f.close();
    if (err) err->clear();
    return true;
}

}
