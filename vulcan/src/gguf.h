#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gguf {

enum GGMLType : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_Q8_0 = 8,
};

struct TensorInfo {
    std::string name;
    uint32_t type = GGML_TYPE_F32;
    std::vector<uint32_t> dims; // shape, reversed (gguf: fastest dim first)
    std::vector<float> data;
};

struct GGUFMeta {
    std::vector<std::pair<std::string, std::string>> strings;
    std::vector<std::pair<std::string, uint32_t>> u32s;
    std::vector<std::pair<std::string, float>> f32s;
};

bool writeGGUF(const std::string& path,
               const std::string& arch,
               const GGUFMeta& meta,
               const std::vector<TensorInfo>& tensors,
               std::string* err);

}
