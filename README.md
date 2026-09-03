# Vulcan AI

**An experiment in building a GPU-native AI training engine with Vulkan — with the long-term goal of outperforming CUDA for training.**

Vulcan AI started with a simple question:

> **Can Vulkan be used to actually train an AI model?**

That question has evolved into a much more ambitious one:

> **Can a purpose-built Vulkan compute engine eventually train AI faster than CUDA?**

Instead of building the core training system on top of an existing machine-learning framework, Vulcan explores what happens when neural-network computation is implemented directly with **Vulkan compute shaders**.

The current project has progressed from a basic experiment into an end-to-end GPU training system capable of training a small **BitNet-style W1.58A8 GPT-2-like transformer**. The system implements tensor operations, forward propagation, backpropagation, quantization, optimization, and CPU/GPU numerical validation.

Vulcan is still experimental. It does **not currently claim to be faster than CUDA**. The goal is to build, measure, optimize, and benchmark the system until that question can be answered with reproducible data.

---

## The Experiment

The project is built around two fundamental questions.

### Can Vulkan train AI?

The first goal was simply to prove that a neural network could be trained using Vulkan compute rather than relying on a traditional machine-learning backend.

That required implementing the basic training pipeline:

```text
Tensor Operations
       ↓
Vulkan Compute Shaders
       ↓
Forward Pass
       ↓
Loss
       ↓
Backward Pass
       ↓
Gradients
       ↓
Optimizer
       ↓
Updated Model
```

Vulcan now has those components working together in an actual training loop.

### Can Vulkan beat CUDA?

Once GPU training was possible, the next question became performance.

CUDA has an enormous ecosystem of highly optimized machine-learning libraries and kernels. Vulcan takes a different approach: build a lightweight training system directly around Vulkan compute and optimize the entire workload for the GPU.

The long-term objective is to determine whether that approach can achieve **higher training throughput than CUDA** on comparable hardware and workloads.

The ultimate goal is therefore not simply:

> **“Vulkan can train a model.”**

It is:

> **“Vulkan can train a model faster.”**

---

# What Vulcan Is

Vulcan is a C++17 machine-learning engine built around Vulkan compute.

The host application constructs a declarative tensor graph while Vulkan compute shaders perform the actual numerical work on the GPU.

```text
C++
 │
 ├── Tensor / Compute Graph
 │
 └── Vulkan Backend
       │
       └── GLSL Compute Shaders
              │
              └── GPU
```

The current implementation focuses on a small BitNet-style transformer because a compact model provides a controlled environment for experimenting with GPU kernels, quantization, numerical correctness, and training performance.

---

# Current Model

The main implementation is:

```text
vulcan_bitnet_gpt
```

It implements a small GPT-2-style transformer using BitNet-inspired **W1.58A8** linear layers.

The current model configuration is:

| Parameter          | Value |
| ------------------ | ----: |
| Hidden size        |    64 |
| Attention heads    |     4 |
| Head dimension     |    16 |
| Feed-forward size  |   256 |
| Transformer blocks |     2 |
| Batch size         |     4 |
| Sequence length    |   785 |
| Input bytes        |   784 |

The model uses learned token and positional embeddings, transformer blocks, causal self-attention, RMSNorm, BitLinear layers, GELU, residual connections, a final RMSNorm, and an untied BitLinear language-model head.

Each input sample is represented as up to **784 UTF-8 bytes**, followed by a separator/classification position.

---

# Transformer Architecture

A transformer block follows this general structure:

```text
                 Input
                   │
                RMSNorm
                   │
          ┌────────┼────────┐
          ↓        ↓        ↓
       BitLinear BitLinear BitLinear
           Q         K        V
          └────────┬────────┘
                   │
            Causal Attention
                   │
               BitLinear
                   │
                Residual
                   │
                RMSNorm
                   │
               BitLinear
                   │
                 GELU
                   │
               BitLinear
                   │
                Residual
```

The final transformer output is normalized and passed through the language-model head to produce logits.

---

# Vulkan Compute Engine

Vulcan represents computation as a **declarative tensor graph**.

A tensor contains CPU-side data and a Vulkan storage buffer. Operations are recorded in the graph and each operation maps to a corresponding GLSL compute shader.

The graph supports separate forward and backward execution while handling tensor uploads and downloads between the CPU and GPU.

Current operations include:

* matrix multiplication
* addition and multiplication
* embeddings
* RMSNorm
* LayerNorm
* softmax
* causal-attention operations
* RoPE
* GELU
* SiLU
* ReLU
* tanh
* cross-entropy
* MSE
* quantization
* row reductions and scaling
* Adam optimization

The compute kernels live in:

```text
vulcan/shaders/
```

---

# Automatic Shader Compilation

Vulcan's CMake configuration automatically discovers every `.comp` file in the shader directory and compiles it to SPIR-V using `glslc`.

For example:

```text
vulcan/shaders/matmul.comp
        ↓
vulcan/build/spv/matmul.spv
```

New shaders are automatically included in the build without requiring a new manual `glslc` command.

---

# BitNet Quantization

The current transformer uses BitNet-style quantization for its linear layers.

## W1.58 Weights

Weights are quantized to ternary values:

```text
-1
 0
+1
```

using a per-output-row scale.

This provides the **W1.58** component of the model.

## A8 Activations

Activations use per-row absolute-maximum quantization into an 8-bit signed range.

The resulting computation follows the general pattern:

```text
Activation
   ↓
A8 Quantization
   ↓
Ternary Weight Quantization
   ↓
Matrix Multiplication
   ↓
Scale
```

The CPU reference implementation mirrors the same quantization procedure used by the GPU implementation.

---

# Low-Precision Compute

In addition to BitNet's own W1.58/A8 quantization, Vulcan supports selectable precision for intermediate tensors in the training graph.

```bash
--fp fp32
--fp fp8
--fp fp4
```

### FP32

Standard float32 computation and the default mode.

### FP8

Uses an E4M3-style floating-point representation.

### FP4

Uses an E2M1 representation with the positive value grid:

```text
0
0.5
1
1.5
2
3
4
6
```

The CPU oracle and GPU graph apply the same precision quantization at matching graph boundaries so low-precision execution can still be checked against the reference implementation.

---

# CPU/GPU Oracle Validation

A major part of the experiment is ensuring that GPU optimizations do not silently produce incorrect training results.

Vulcan therefore maintains a CPU implementation of the model as a reference or **oracle**.

Every GPU training step is compared against the CPU implementation before the optimizer is allowed to update the model.

The current acceptance criteria are:

```text
loss_diff    < 5e-3
grad_maxdiff < 1e-3
```

When the GPU and CPU results do not agree within those limits, the training process aborts with an **ORACLE FAIL** rather than continuing with potentially incorrect gradients.

This is particularly useful for performance experimentation because new kernels can be aggressively optimized while still being checked against a known reference implementation.

---

# Performance Goal

The long-term goal of Vulcan is **training performance**.

The project is exploring whether a carefully designed Vulkan training engine can eventually outperform CUDA-based implementations for comparable workloads.

That requires optimizing much more than individual shaders.

Important areas include:

```text
Kernel throughput
Memory bandwidth
Tensor layout
Kernel fusion
Synchronization
GPU utilization
CPU/GPU communication
Memory reuse
Launch overhead
Attention performance
Matrix multiplication
Low-precision arithmetic
```

The project therefore treats benchmarking as part of the experiment itself.

A meaningful CUDA comparison should use equivalent:

* hardware
* model architecture
* batch size
* precision
* workload
* dataset
* optimizer configuration

and should report reproducible measurements such as:

```text
Samples / second
Tokens / second
Training step time
GPU utilization
Memory usage
```

Until such benchmarks demonstrate otherwise, **Vulcan should be considered an experimental system pursuing the goal of beating CUDA, not one that has already achieved it.**

---

# Dataset Pipeline

Vulcan includes a dataset streaming tool:

```text
vulcan/tools/stream_dataset.py
```

The tool downloads dataset samples from the Hugging Face Hub using streaming mode and converts them into Vulcan's compact `.bin` format.

For example:

```bash
python vulcan/tools/stream_dataset.py \
    bigcode/starcoderdata \
    code \
    --dataset-kwarg data_dir=python \
    --text-col content \
    --take 1000 \
    --split-test 200
```

Conversational datasets can also be processed:

```bash
python vulcan/tools/stream_dataset.py \
    WithinUsAI/claude_mythos_distilled_25k \
    mythos \
    --messages \
    --label-col category \
    --take 1000 \
    --split-test 100
```

The pipeline deliberately uses streaming and a configurable row limit so large datasets do not have to be downloaded wholesale.

---

# Dataset Format

The current `.bin` format is intentionally simple:

```text
uint32 n
uint32 feat
n × feat float32 values
n × float32 labels
```

For the current byte-level model:

```text
feat = 784
```

Each input row contains up to 784 bytes of UTF-8 encoded text and is padded with zero bytes when necessary.

Labels are stored separately and are used by the classification portion of the current model.

---

# Training

Build the project with:

```bash
cmake -B vulcan/build -G Ninja
cmake --build vulcan/build
```

The primary trainer can then be run with:

```bash
vulcan/build/vulcan_bitnet_gpt \
    <max_train> \
    <epochs> \
    <seed> \
    <base> \
    [--fp fp32|fp8|fp4]
```

For example:

```bash
vulcan/build/vulcan_bitnet_gpt 1000 1 42 code --fp fp8
```

The trainer reads:

```text
vulcan/data/<base>_train.bin
vulcan/data/<base>_test.bin
```

and exports:

```text
vulcan/data/<base>-bitnet-gpt2-<precision>.gguf
```

---

# GGUF Export

Vulcan contains a minimal GGUF writer used by the trainer to serialize model weights.

The exporter currently produces GGUF v3 files and supports the tensor metadata required by the current BitNet model export path. The current implementation emits F32 tensors.

Example:

```text
code-bitnet-gpt2-fp32.gguf
```

---

# Project Structure

```text
.
├── README.md
├── AGENTS.md
└── vulcan/
    ├── CMakeLists.txt
    │
    ├── src/
    │   ├── vulkan_api.{h,cpp}
    │   ├── graph.{h,cpp}
    │   ├── context.{h,cpp}
    │   ├── precision.h
    │   ├── gguf.{h,cpp}
    │   ├── bitnet_gpt.cpp
    │   ├── bitnet.cpp
    │   └── gpt2.cpp
    │
    ├── shaders/
    │   └── *.comp
    │
    ├── tools/
    │   └── stream_dataset.py
    │
    └── llama.cpp/
```

The current CMake project builds three model targets:

```text
vulcan_bitnet_gpt
vulcan_bitnet
vulcan_gpt
```

Of these, `vulcan_bitnet_gpt` is the primary working trainer. The other two targets are currently stubs rather than complete training implementations.

---

# Requirements

* Vulkan SDK
* `glslc` on `PATH`
* CMake 3.20+
* Ninja
* C++17 compiler
* Python 3 for the dataset tooling

---

# Building

From the repository root:

```bash
cmake -B vulcan/build -G Ninja
cmake --build vulcan/build
```

Compiled shaders will be placed in:

```text
vulcan/build/spv/
```

The Vulkan backend and shader system are built directly through the project's CMake configuration.

---

# Why Build This?

Most machine-learning development happens through increasingly high-level software stacks.

Vulcan deliberately moves in the opposite direction.

The project asks what happens when the machine-learning training loop is built much closer to the GPU:

```text
Model
  ↓
Tensor Graph
  ↓
Vulkan
  ↓
Compute Shaders
  ↓
GPU
```

That makes it possible to experiment directly with:

* tensor representation
* GPU memory movement
* kernel design
* numerical precision
* quantization
* synchronization
* scheduling
* optimization
* transformer implementation
* training throughput

The point is not to recreate every feature of a mature ML framework.

The point is to discover how capable a purpose-built Vulkan training engine can become.

---

# Long-Term Direction

The long-term vision is to evolve Vulcan from an experimental proof of concept into a highly optimized GPU training engine.

Potential areas of future work include:

* fused kernels
* improved matrix multiplication
* optimized attention kernels
* reduced synchronization
* GPU-resident optimizer state
* improved tensor layouts
* better memory reuse
* asynchronous execution
* more aggressive low-precision computation
* larger models
* multi-GPU training
* broader model support
* systematic CUDA benchmarking

The ultimate question remains:

> **Can Vulkan become a faster foundation for AI training than CUDA?**

Vulcan exists to find out.

---

# Project Status

**Experimental / Research**

Vulcan has already demonstrated the core idea: **Vulkan can be used to train an AI model.**

The project is now focused on the harder part:

**How fast can it become?**

The intended progression is:

```text
Can Vulkan train AI?
        ↓
Yes.
        ↓
Can Vulkan train useful models?
        ↓
Can Vulkan train them efficiently?
        ↓
Can Vulkan outperform CUDA?
```

The final answer to the last question has not been established yet.

**The experiment is still running.**

---

# Contributing

Contributions are welcome, particularly in areas such as:

* Vulkan compute optimization
* transformer kernels
* matrix multiplication
* attention
* quantization
* numerical validation
* memory management
* benchmarking
* portability
* model architectures
* dataset tooling

When adding a new compute operation, the corresponding GLSL shader should be placed in:

```text
vulcan/shaders/
```

and wired into the graph/backend.

---

# License

See the repository's license information for the terms applicable to this project.
