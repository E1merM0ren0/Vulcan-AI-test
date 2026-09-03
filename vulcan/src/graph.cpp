// graph.cpp — Execute a Tensor/Op graph on the Vulkan compute backend.
//
// runOp() is the workhorse: it maps every OpKind to a SPIR-V shader name,
// chooses a tensor-binding order (most ops use {a,b,c,d,e,out}; some special
// ops like QUANTIZE_FP and NORM_BWD permute the slots), computes the
// workgroup dispatch size, and issues a single runCompute() call.
//
// The Tensor::dirtyUpload / dirtyDownload flags are kept in sync:
//   uploadAll()   – pushes all tensors that were modified on the CPU
//   forward()     – runs forward ops, marks all tensors dirtyDownload
//   downloadAll() – pulls results back to CPU for the oracle check
//   backward()    – same pattern, for backward ops

#include "graph.h"
#include "vulkan_api.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

struct ShaderSpec {
    const char* name;
    OpKind kind;
};

static const char* shaderFor(OpKind kind) {
    switch (kind) {
        case OP_ZERO: return "zero";
        case OP_COPY: return "copy";
        case OP_ADD: return "add";
        case OP_MUL: return "mul";
        case OP_MUL_BWD: return "mul_bwd";
        case OP_MATMUL: return "matmul";
        case OP_RMS_NORM: return "rms_norm";
        case OP_RMS_NORM_BWD: return "rms_norm_bwd";
        case OP_LAYER_NORM: return "layer_norm";
        case OP_LAYER_NORM_BWD: return "layer_norm_bwd";
        case OP_QUANTIZE_ABSMAX: return "quantize_absmax";
        case OP_QUANTIZE_TERNARY: return "quantize_ternary";
        case OP_SCALE_ROWS: return "scale_rows";
        case OP_GELU: return "gelu";
        case OP_GELU_BWD: return "gelu_bwd";
        case OP_TANH: return "tanh";
        case OP_TANH_BWD: return "tanh_bwd";
        case OP_SILU: return "silu";
        case OP_SILU_BWD: return "silu_bwd";
        case OP_RELU: return "relu";
        case OP_RELU_BWD: return "relu_bwd";
        case OP_CROSS_ENTROPY: return "cross_entropy";
        case OP_CROSS_ENTROPY_BWD: return "cross_entropy_bwd";
        case OP_SOFTMAX: return "softmax";
        case OP_SOFTMAX_BWD: return "softmax_bwd";
        case OP_ADAM: return "adam";
        case OP_ADD_POS: return "add_pos";
        case OP_ADD_POS_BWD: return "add_pos_bwd";
        case OP_ADD_BIAS: return "add_bias";
        case OP_EMBEDDING: return "embedding";
        case OP_EMBEDDING_BWD: return "embedding_bwd";
        case OP_REDUCE_SUM_ROWS: return "reduce_sum_rows";
        case OP_SPLIT_HEADS: return "split_heads";
        case OP_MERGE_HEADS: return "merge_heads";
        case OP_MSE_SQUARE: return "mse_square";
        case OP_MSE_BWD: return "mse_bwd";
        case OP_ROPE: return "rope";
        case OP_QUANTIZE_FP: return "quantize_fp";
    }
    return "zero";
}

Tensor* Graph::alloc(int n, int cols) {
    Tensor* t = new Tensor();
    t->n = n;
    t->cols = cols;
    t->data.assign((size_t)n * cols, 0.0f);
    t->dirtyUpload = true;
    t->dirtyDownload = false;
    tensors.push_back(t);
    return t;
}

void Graph::add(const Op& op) {
    ops.push_back(op);
}

static void ensureBuffer(Tensor* t, vk::Context* ctx) {
    if (!t) return;
    VkDeviceSize needed = (VkDeviceSize)t->n * (VkDeviceSize)t->cols * sizeof(float);
    if (t->buf == VK_NULL_HANDLE || t->bufSize < needed) {
        if (t->buf != VK_NULL_HANDLE) ctx->destroyBuffer(t->buf);
        t->buf = ctx->createBuffer(needed);
        t->bufSize = needed;
    }
}

void Graph::uploadAll() {
    for (Tensor* t : tensors) {
        if (!t || !t->dirtyUpload) continue;
        if (t->n * t->cols > 0) {
            ensureBuffer(t, ctx);
            ctx->uploadBuffer(t->buf, t->bufSize, t->data.data());
        }
        t->dirtyUpload = false;
    }
}

void Graph::downloadAll() {
    for (Tensor* t : tensors) {
        if (!t || !t->dirtyDownload) continue;
        if (t->buf != VK_NULL_HANDLE && t->n * t->cols > 0) {
            ctx->downloadBuffer(t->buf, t->bufSize, t->data.data());
        }
        t->dirtyDownload = false;
    }
}

// Reinterpret a float as a raw uint32 bit pattern (for passing floats through
// int32 push constants).
static uint32_t f32bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// Round up a/b to the next integer (for computing workgroup dispatch counts).
static uint32_t ceilDiv(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// Dispatch a single op: resolve its shader + binding order, pick a dispatch
// size, and call runCompute(). Returns false if the shader could not be loaded.
//
// Binding-order permutations (because a few shaders put operands at unusual
// descriptor slots):
//   default  {a,b,c,d,e,out}
//   normBwd  {a,b,out,d,e}->{0,1,2,5,3,4}  (d = reduction temp)
//   precFwd  {a,out,b,c,d,e}->{0,5,1,2,3,4}  (QUANTIZE_FP is in-place: a==out)
static bool runOp(vk::Context* ctx, const Op& op) {
    Tensor* ts[6] = {op.a, op.b, op.c, op.d, op.e, op.out};
    const int* order = nullptr;
    static const int def[6] = {0, 1, 2, 3, 4, 5};
    static const int normBwd[6] = {0, 1, 2, 5, 3, 4};  // a, b, c, out, d, e
    static const int precFwd[6] = {0, 5, 1, 2, 3, 4};  // a, out, b, c, d, e
    if (op.kind == OP_RMS_NORM_BWD || op.kind == OP_LAYER_NORM_BWD) order = normBwd;
    else if (op.kind == OP_QUANTIZE_FP) order = precFwd;
    else order = def;
    std::vector<VkBuffer> bufs;
    for (int i = 0; i < 6; i++) {
        Tensor* t = ts[order[i]];
        if (t && t->buf != VK_NULL_HANDLE) {
            bufs.push_back(t->buf);
        }
    }
    if (bufs.empty()) return true;

    uint32_t pc[8];
    std::memcpy(pc, op.pc, sizeof(pc));

    uint32_t n = 0, cols = 0, w1 = 1, w2 = 1, w3 = 1;
    const char* sh = shaderFor(op.kind);

    switch (op.kind) {
        case OP_ZERO:
            pc[0] = (uint32_t)(op.out ? op.out->n * op.out->cols : 0);
            w1 = ceilDiv(pc[0], 64);
            break;
        case OP_COPY:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_ADD:
        case OP_MUL:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_MUL_BWD:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_MATMUL: {
            // C[m,nn] = A[m,k] * B[k,nn].
            // pc[3] (tA) = transpose A (A is stored k×m); pc[4] (tB) similarly.
            uint32_t tA = pc[3], tB = pc[4];
            uint32_t m = tA ? (uint32_t)op.a->cols : (uint32_t)op.a->n;
            uint32_t k = tA ? (uint32_t)op.a->n : (uint32_t)op.a->cols;
            uint32_t nn = tB ? (uint32_t)op.b->n : (uint32_t)op.b->cols;
            if (tB) k = (uint32_t)op.b->cols;
            pc[0] = m; pc[1] = nn; pc[2] = k;
            w1 = ceilDiv(m, 16); w2 = ceilDiv(nn, 16);
            break;
        }
        case OP_RMS_NORM:
        case OP_RMS_NORM_BWD:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 256);
            break;
        case OP_LAYER_NORM:
        case OP_LAYER_NORM_BWD:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 256);
            break;
        case OP_QUANTIZE_ABSMAX:
        case OP_QUANTIZE_TERNARY:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 256);
            break;
        case OP_SCALE_ROWS:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_GELU:
        case OP_TANH:
        case OP_SILU:
        case OP_RELU:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_GELU_BWD:
        case OP_TANH_BWD:
        case OP_SILU_BWD:
        case OP_RELU_BWD:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_CROSS_ENTROPY:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n, 64);
            break;
        case OP_CROSS_ENTROPY_BWD:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 256);
            break;
        case OP_SOFTMAX:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_SOFTMAX_BWD:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_ADAM:
            pc[0] = (uint32_t)(op.a->n * op.a->cols);
            w1 = ceilDiv(pc[0], 64);
            break;
        case OP_ADD_POS:
        case OP_ADD_POS_BWD:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_ADD_BIAS:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_EMBEDDING:
            n = (uint32_t)op.b->n; cols = (uint32_t)op.b->cols;
            pc[0] = (uint32_t)op.a->n; pc[1] = cols;
            w1 = ceilDiv(pc[0] * cols, 64);
            break;
        case OP_EMBEDDING_BWD:
            n = (uint32_t)op.b->n; cols = (uint32_t)op.b->cols;
            pc[0] = (uint32_t)op.a->n; pc[1] = cols; pc[2] = (uint32_t)op.pc[2];
            w1 = ceilDiv(pc[2] * cols, 64);
            break;
        case OP_REDUCE_SUM_ROWS:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(cols, 64);
            break;
        case OP_SPLIT_HEADS:
        case OP_MERGE_HEADS:
            w1 = ceilDiv((uint32_t)op.out->n * (uint32_t)op.out->cols, 256);
            break;
        case OP_MSE_SQUARE:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n, 64);
            break;
        case OP_MSE_BWD:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 64);
            break;
        case OP_ROPE:
            n = (uint32_t)op.a->n; cols = (uint32_t)op.a->cols;
            pc[0] = n; pc[1] = cols;
            w1 = ceilDiv(n * cols, 256);
            break;
        case OP_QUANTIZE_FP:
            // In-place re-quantization to fp8/fp4 at compute-graph boundaries
            // (mirrors the CPU oracle's q()). pc[0] = element count; pc[1] = mode
            // (1=fp8/E4M3, 2=fp4/E2M1). Operand order is {a,out,b,c,d,e}.
            pc[0] = (uint32_t)(op.a->n * op.a->cols);
            pc[1] = (uint32_t)op.pc[1];  // mode: 1=fp8, 2=fp4
            w1 = ceilDiv(pc[0], 256);
            break;
    }

    return ctx->runCompute(sh, w1, w2, w3, bufs, pc, 32);
}

// Replay every forward (bwd==false) op, then download all results so the CPU
// oracle can read them. Dirty-tracking is coarse: we mark everything dirty
// after a pass and download only what had a buffer.
void Graph::forward() {
    uploadAll();
    for (Op& op : ops) {
        if (op.bwd) continue;
        if (!runOp(ctx, op)) {
            std::cerr << "op failed (forward)\n";
            break;
        }
    }
    for (Tensor* t : tensors) t->dirtyDownload = true;
    downloadAll();
}

// Mirrors forward() but for backward (bwd==true) ops, run in list order.
void Graph::backward() {
    uploadAll();
    for (Op& op : ops) {
        if (!op.bwd) continue;
        if (!runOp(ctx, op)) {
            std::cerr << "op failed (backward)\n";
            break;
        }
    }
    for (Tensor* t : tensors) t->dirtyDownload = true;
    downloadAll();
}
