# Vulcan AI

A Vulkan-compute machine-learning engine with small C++ trainers. It implements
BitNet-style (W1.58A8) GPT-2 language models that train **on the GPU** while
being continuously validated against an exact **CPU oracle** — every training
step's loss and gradients must match the reference CPU implementation inside a
tight tolerance, or training aborts.

All heavy compute (embedding, RMSNorm, BitLinear matmuls, causal attention,
Adam) runs as small GLSL compute shaders; the host C++ builds a declarative
tensor graph and dispatches it through a minimal Vulkan wrapper.

> This repo is a **partial recovery after data loss**. Some trainers' source
> files and original data-pipeline scripts are gone. See
> [Recovery notes](#recovery-notes).

---

## Components

| Path | Purpose |
|------|---------|
| `vulcan/src/vulkan_api.{h,cpp}` | Minimal Vulkan compute wrapper (device, buffers, pipeline cache, dispatch) |
| `vulcan/src/graph.{h,cpp}` | Declarative tensor/op compute graph + shader dispatch |
| `vulcan/src/precision.h` | FP8 (E4M3) / FP4 (E2M1) quantization shared CPU+GPU (`--fp` knob) |
| `vulcan/src/context.{h,cpp}` | `.bin` dataset reader/writer + small helpers |
| `vulcan/src/gguf.{h,cpp}` | Minimal GGUF (llama.cpp) model writer |
| `vulcan/src/bitnet_gpt.cpp` | The main trainer (`vulcan_bitnet_gpt`) |
| `vulcan/src/bitnet.cpp`, `gpt2.cpp` | Stub trainers (`vulcan_bitnet`, `vulcan_gpt`) |
| `vulcan/shaders/*.comp` | GLSL compute kernels (auto-compiled to `.spv`) |
| `vulcan/tools/stream_dataset.py` | Stream any Hugging Face dataset into `.bin` |
| `vulcan/llama.cpp/` | Vendored upstream llama.cpp (reference only — see note below) |
| `vulcan/vulcanref/` | A Cython/Python module wrapping parts of the engine (unused by the trainer) |

---

## Requirements

- **Vulkan SDK** (with `glslc` on `PATH`)
- **CMake >= 3.20**
- **Ninja**
- A C++17 compiler (tested with the `w64devkit` toolchain)
- Python 3 (optional — only for the data-pipeline script)

---

## Build

```bash
cmake -B vulcan/build -G Ninja
cmake --build vulcan/build
```

Outputs land in `vulcan/build/`:

- `vulcan_bitnet_gpt` — the working BitNet-GPT-2 trainer
- `vulcan_bitnet`, `vulcan_gpt` — stub trainers (not yet reimplemented)
- `vulcan/build/spv/*.spv` — compiled shaders

**Adding a new shader:** just drop `vulcan/shaders/<name>.comp`. The build
glob (`CONFIGURE_DEPENDS`) compiles it to `<name>.spv` automatically — there is
no manual `glslc` line to add.

> **Runtime SPV note:** the trainer looks for compiled shaders at
> `vulcan/build/spv` (see `findSPVDir()`), *relative to the current working
> directory*. If you build into a fresh temp dir and run from the repo root, a
> newly added shader won't be found there — copy its `.spv` into
> `vulcan/build/spv/` (or run from the build directory).

---

## Training

```bash
vulcan/build/vulcan_bitnet_gpt <max_train> <epochs> <seed> <base> [--fp fp32|fp8|fp4]
```

- Reads `vulcan/data/<base>_train.bin` / `<base>_test.bin`
- Exports `vulcan/data/<base>-bitnet-gpt2-<fp>.gguf`
- Arguments are validated (non-numeric → usage + exit `2`); `--help` prints usage

### Compute precision (`--fp`)

| Value | Meaning |
|-------|---------|
| `fp32` (default) | Full float32 compute — unchanged behaviour |
| `fp8` | Re-quantize every intermediate/gradient to E4M3 at each graph boundary |
| `fp4` | Re-quantize to E2M1 (grid `0, 0.5, 1, 1.5, 2, 3, 4, 6`) |

`fp8`/`fp4` re-quantize every intermediate and gradient tensor **at the exact
same graph boundaries on both the CPU oracle and the GPU**, so the oracle gate
stays exact (`loss_diff=0, grad_diff=0`) while the real compute runs in low
precision. Bitnet's own weight/activation quantization (W1.58 / A8 absmax) is
**unchanged**.

#### Oracle gate

Each GPU step is compared against the CPU oracle:

```
loss_diff < 5e-3   and   pre-Adam grad maxdiff < 1e-3   →  apply Adam
any mismatch                                              →  ORACLE FAIL, abort
```

### Known dataset bases

`mythos`, `sumtables`, `fable5`, `code`, `mnist` (look in `vulcan/data/`).

### `.bin` format

`uint32 n` · `uint32 feat` (= 784) · `n×784` float32 features (each 0–255) ·
`n` float32 labels (class id). Rows are the UTF-8 bytes of text, right-padded
with `0`.

---

## Data pipeline

`.bin` files are generated from Hugging Face datasets by a streaming script
(this is the recovery replacement for the original per-dataset `tools/stream_*`
scripts, which were lost):

```bash
python vulcan/tools/stream_dataset.py <hf_repo> <base> \
    --text-col CONTENT [--label-col LABEL] \
    [--take 1000] [--split-test N] [--dataset-kwarg data_dir=python] [--messages]
```

- `BYTES = 784` in the streamer **must** match `BYTES` in `bitnet_gpt.cpp`.
- **Storage is tight:** stream and cap at the first ~1000 rows with `--take`;
  never bulk-clone a dataset.
- All candidate datasets are on **Hugging Face**, not GitHub; `git clone` of
  their GitHub mirror URLs does not work.

Suggested candidates (see `AGENTS.md`): `bigcode/starcoderdata` (python subset)
for pretraining; `WithinUsAI/claude_mythos_distilled_25k` (SFT, chat) and the
`algocean-fable5-traces` / `TRACCERR/Sumtables-...` traces for alignment.

---

## How it works (architecture)

1. **Vulkan wrapper** (`vulkan_api`) owns the device and one compute queue. Every
   tensor is a host-visible/coherent storage buffer (persistent mapping), so
   upload/download is a plain `memcpy` and results are visible after `vkQueueWaitIdle`.
2. **Graph** (`graph`) is a flat list of **tensors** + **ops**. Each op maps to
   one GLSL shader with up to 5 tensor operands + an output, push constants, and
   a `bwd` flag. `Graph::forward()` replays forward ops; `Graph::backward()`
   replays backward ops; dirty-flag tracking uploads/downloads only what changed.
3. **Trainer** (`bitnet_gpt`) builds the graph with `precQ()` precision-injection
   ops, wires the CPU oracle mirror, runs the oracle-gated loop, then exports GGUF.

---

## Git / backup

- Remote: `origin` → `https://github.com/E1merM0ren0/Vulcan-AI-test.git` (`main`).
- Push is the backup workflow: `git commit -am "<msg>" && git push origin main`.
  If the push is rejected, `git pull --rebase origin main` first.

**Gitignored (never commit):** `vulcan/build/`, `vulcan/data/*.bin|*.gguf`,
`vulcan/llama.cpp/` (vendored), root `data/`, `*.pyc/.pyd`, root `*.out` logs.

---

## Recovery notes

- The original per-dataset `tools/stream_*.py` scripts are gone; use
  `tools/stream_dataset.py`.
- `vulcan/CMakeLists.txt` builds only 3 models, but `vulcan/build/` may contain
  stale binaries/logs for trainers whose sources were lost (`vulcan_bert`,
  `vulcan_codelang`, `vulcan_falcon`, `vulcan_gptneox`, `vulcan_llama`,
  `vulcan_train`) — those are dead binaries, ignore them.
- Root `llama.cpp/` is an **empty leftover git repo** (only `.git`, no commits).
  Never `git add` it — git would record a broken submodule link.
- `vulcan/llama.cpp/` is a vendored upstream reference for GGUF/mllama compatibility.
