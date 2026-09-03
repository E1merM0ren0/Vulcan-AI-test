// graph.h — Declarative GPU compute graph over storage-buffer tensors.
//
// A Tensor owns a CPU-side std::vector<float> plus a Vulkan storage buffer
// handle. An Op records one shader dispatch (an OpKind, 5 optional tensor
// operands, an output, push constants, and a fwd/bwd flag). A Graph holds a
// flat list of tensors + ops and replays them in order via the Vulkan context.
//
// The graph is NOT general-purpose: every Op kind maps to exactly one GLSL
// shader (see graph.cpp shaderFor()), and the tensor roles are implicit in the
// shader's binding layout. Push constants are stored as raw int32 bits and
// reinterpret-cast inside runOp to communicate shape / hyperparameters.
//
// Forward (bwd=false) and backward (bwd=true) ops are interleaved in the same
// list.  Graph::forward() replays only the forward ops; Graph::backward()
// replays only the backward ops. After each pass every tensor is downloaded
// back into its CPU-side data[] buffer.

#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace vk {
class Context;
}

struct Tensor {
    int n = 0, cols = 0;      // logical shape: n rows, cols columns (n*cols floats)
    std::vector<float> data;  // CPU-side buffer (uploaded to GPU before dispatch)
    VkBuffer buf = VK_NULL_HANDLE;    // Vulkan storage-buffer handle (lazily allocated)
    VkDeviceSize bufSize = 0;          // byte size of the allocated buffer (may exceed n*cols*4)
    bool dirtyUpload = true;   // true when data[] has changed since last upload
    bool dirtyDownload = false; // true when the GPU wrote new values not yet downloaded
};

// Ordered list of GPU shader operations. Each entry maps 1:1 to a GLSL
// shader in vulcan/shaders/<name>.comp and is dispatched by graph.cpp runOp().
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

// One op node in the graph. The tensor roles (a/b/c/d/e/out) are determined
// by the shader and each op kind expects a specific number of them; unused
// tensor slots are left null.  pc[0..7] carry push constants (shape dims,
// encoding flags, ln-epsilon as bits, etc.).
struct Op {
    OpKind kind = OP_ZERO;
    Tensor* a = nullptr;   // most ops: primary input / weight
    Tensor* b = nullptr;   // most ops: secondary input / bias / scale
    Tensor* c = nullptr;   // unused by most ops
    Tensor* d = nullptr;   // unused by most ops
    Tensor* e = nullptr;   // unused by most ops
    Tensor* out = nullptr; // output tensor (may alias a for in-place ops)
    int32_t pc[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // push constants, shader-specific
    bool bwd = false;      // true = backward-pass op (skipped in forward())
    bool inPlace = false;  // (unused in current graphs; kept for API stability)
};

// The complete compute graph: owns all tensors, records all ops, and replays
// them through a Vulkan context.  Upload/download is automatic: uploadAll()
// pushes every dirty-upload tensor to the GPU; downloadAll() pulls back every
// dirty-download tensor after dispatch.
struct Graph {
    std::vector<Tensor*> tensors;  // all allocated tensors (owned, freed by caller or shutdown)
    std::vector<Op> ops;           // ordered list of forward then backward ops
    vk::Context* ctx = nullptr;    // Vulkan backend (not owned)
    int spvDirIndex = 0;           // (unused placeholder)

    // Allocate a zeroed n×cols tensor tracked by this graph.
    Tensor* alloc(int n, int cols);
    // Append an op to the graph (call order matters: forward ops first, backward ops after).
    void add(const Op& op);
    // Replay all non-backward ops (upload → dispatch forward shaders → download).
    void forward();
    // Replay all backward ops (upload → dispatch backward shaders → download).
    void backward();
    // Upload every tensor whose dirtyUpload flag is set.
    void uploadAll();
    // Download every tensor whose dirtyDownload flag is set.
    void downloadAll();
};

// Shorthand for graph.alloc(n,cols).
Tensor* allocTensor(Graph& g, int n, int cols);

// Reserved entry point for a future full-graph constructor. Currently only
// declared (never defined, never called); the actual graph is built inline by
// GPUModel::build() in bitnet_gpt.cpp.
Graph& buildGraph();
