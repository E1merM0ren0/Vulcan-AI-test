#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace vk {
class Context;
}

struct Tensor {
    int n = 0, cols = 0;
    std::vector<float> data;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize bufSize = 0;
    bool dirtyUpload = true;
    bool dirtyDownload = false;
};

enum OpKind : int {
    OP_ZERO = 0,
    OP_COPY,
    OP_ADD,
    OP_MUL,
    OP_MUL_BWD,
    OP_MATMUL,
    OP_RMS_NORM,
    OP_RMS_NORM_BWD,
    OP_LAYER_NORM,
    OP_LAYER_NORM_BWD,
    OP_QUANTIZE_ABSMAX,
    OP_QUANTIZE_TERNARY,
    OP_SCALE_ROWS,
    OP_GELU,
    OP_GELU_BWD,
    OP_TANH,
    OP_TANH_BWD,
    OP_SILU,
    OP_SILU_BWD,
    OP_RELU,
    OP_RELU_BWD,
    OP_CROSS_ENTROPY,
    OP_CROSS_ENTROPY_BWD,
    OP_SOFTMAX,
    OP_SOFTMAX_BWD,
    OP_ADAM,
    OP_ADD_POS,
    OP_ADD_POS_BWD,
    OP_ADD_BIAS,
    OP_EMBEDDING,
    OP_EMBEDDING_BWD,
    OP_REDUCE_SUM_ROWS,
    OP_SPLIT_HEADS,
    OP_MERGE_HEADS,
    OP_MSE_SQUARE,
    OP_MSE_BWD,
    OP_ROPE,
    OP_QUANTIZE_FP,
};

struct Op {
    OpKind kind = OP_ZERO;
    Tensor* a = nullptr;
    Tensor* b = nullptr;
    Tensor* c = nullptr;
    Tensor* d = nullptr;
    Tensor* e = nullptr;
    Tensor* out = nullptr;
    int32_t pc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool bwd = false;
    bool inPlace = false;
};

struct Graph {
    std::vector<Tensor*> tensors;
    std::vector<Op> ops;
    vk::Context* ctx = nullptr;
    int spvDirIndex = 0;

    Tensor* alloc(int n, int cols);
    void add(const Op& op);
    void forward();
    void backward();
    void uploadAll();
    void downloadAll();
};

Tensor* allocTensor(Graph& g, int n, int cols);

Graph& buildGraph();
