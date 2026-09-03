// gguf.h — Minimal GGUF (llama.cpp model) writer.
//
// Produces a version-3 GGUF container from a flat list of named float tensors
// plus optional string / uint32 / float32 metadata key-values. Handles the
// 32-byte tensor-alignment padding required by the format. Only what the
// bitnet trainer needs is implemented (F32 and Q8_0 types, no quantization
// is actually applied here). See gguf.cpp writeGGUF() for the byte layout.

#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gguf {

// GGML quant-type enum values we support writing. Only F32 is emitted today.
enum GGMLType : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_Q8_0 = 8,
};

// One named tensor to serialize: its shape (fastest dimension first, per GGUF),
// storage type, and the raw float payload (always F32 in memory).
struct TensorInfo {
    std::string name;
    uint32_t type = GGML_TYPE_F32;
    std::vector<uint32_t> dims; // shape, reversed (gguf: fastest dim first)
    std::vector<float> data;
};

// Typed metadata key-value groups written into the GGUF header.
struct GGUFMeta {
    std::vector<std::pair<std::string, std::string>> strings;
    std::vector<std::pair<std::string, uint32_t>> u32s;
    std::vector<std::pair<std::string, float>> f32s;
    std::vector<std::pair<std::string, std::vector<std::string>>> strArrays;
    std::vector<std::pair<std::string, std::vector<int32_t>>> i32Arrays;
};

// Write `tensors` with `meta` under the architecture `arch` to `path`.
// Returns false (and sets *err) if the file cannot be opened.
bool writeGGUF(const std::string& path,
               const std::string& arch,
               const GGUFMeta& meta,
               const std::vector<TensorInfo>& tensors,
               std::string* err);

}
