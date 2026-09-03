# AGENTS.md

## What this is

`vulcan/` is a Vulkan compute engine + small C++ trainers (bitnet, bitnet_gpt, gpt2) with GLSL compute kernels. The repo is a partial recovery after data loss: **some files are missing/corrupted** (see "Recovery caveats"). Pushing to GitHub is the backup workflow.

## Build

- Prereqs: Vulkan SDK, `glslc` on PATH, CMake >= 3.20, Ninja.
- Configure + build: `cmake -B vulcan/build -G Ninja && cmake --build vulcan/build`
- Outputs in `vulcan/build/`: `vulcan_bitnet_gpt`, `vulcan_bitnet`, `vulcan_gpt`.
- New shader: just drop `vulcan/shaders/<name>.comp`; `CMakeLists.txt` auto-globs `shaders/*.comp` → `<name>.spv` at build time (no manual `glslc` lines to add).

## Training

- Run: `vulcan/build/vulcan_bitnet_gpt <max_train> <epochs> <seed> <base> [--fp fp32|fp8|fp4]`
  - reads `vulcan/data/<base>_train.bin` / `<base>_test.bin`
  - exports `vulcan/data/<base>-bitnet-gpt2-<fp>.gguf`
  - args are validated (non-numeric → usage + exit 2); `--help` prints usage
- `--fp` selects compute precision (default `fp32` = unchanged). `fp8` (E4M3) / `fp4` (E2M1) re-quantize every intermediate/gradient tensor in the compute graph at the same boundaries on BOTH the CPU oracle and the GPU, so the oracle gate stays exact (`loss_diff=0, grad_diff=0`). Bitnet weight quantization (W1.58 / A8 absmax) is unchanged. Implemented in `src/precision.h` + `shaders/quantize_fp.comp` (+ `OP_QUANTIZE_FP` in `graph.cpp`).
  - Note: `quantize_fp.spv` must be present in the SPV dir at runtime; the temp/fresh build dirs (e.g. `opencode/vbuild`) are not searched by `findSPVDir()`, only `vulcan/build/spv`.
- Known dataset bases (in `vulcan/data/` + build logs): `mythos`, `sumtables`, `fable5`, `code`, `mnist`.
- Harness runs a GPU-vs-CPU oracle check (`ORACLE: PASS`) before training; past runs are logged in `vulcan/build/*.log`.
- `.bin` format (see `context.cpp` `loadBin`): `uint32 n, uint32 feat(=784)`, then n×784 float32 bytes (each 0..255), then n float32 labels (class id). Rows are the UTF-8 bytes of text, right-padded with 0.

## Data pipeline (partially restored)

- `vulcan/tools/stream_dataset.py` regenerates `.bin` from any HF dataset (streaming, capped at `--take`, default 1000). It is the recovery replacement for the lost `tools/stream_*.py` scripts.
- Usage: `python vulcan/tools/stream_dataset.py <hf_repo> <base> --text-col CONTENT [--label-col LABEL] [--take 1000] [--split-test N] [--dataset-kwarg data_dir=python] [--messages]`
- Must match C++: `BYTES=784` in the streamer must equal `BYTES` in `bitnet_gpt.cpp`.

## Recovery caveats

- Original per-dataset `tools/stream_*.py` scripts are GONE (lost to data loss). `vulcan/tools/stream_dataset.py` is the generic replacement; don't assume the older loader/streamer sources exist.
- `vulcan/CMakeLists.txt` only builds 3 models, but `vulcan/build/` has stale binaries/logs for trainers whose sources were lost (`vulcan_bert`, `vulcan_codelang`, `vulcan_falcon`, `vulcan_gptneox`, `vulcan_llama`, `vulcan_train`). Ignore those.
- Root `llama.cpp/` is an empty leftover git repo (only `.git`, no commits). Never `git add` it — git would record a broken submodule link.

## Git / backup

- Remote: `origin` = https://github.com/E1merM0ren0/Vulcan-AI-test.git (branch `main`).
- Gitignored — never commit: `vulcan/build/`, `vulcan/data/*.bin|*.gguf`, `vulcan/llama.cpp/` (vendored reference), root `data/`, `*.pyc/.pyd`, root `*.out` logs.
- Backup step: `git commit -am "<msg>" && git push origin main`. If push is rejected, `git pull --rebase origin main` first (remote has pushed standalone commits before).

## Datasets — all on Hugging Face, NOT GitHub

The earlier `git clone https://github.com/<repo>` attempts failed because these are **Hugging Face** dataset repos. **Storage is tight: only take the first ~1000 rows per dataset** using HF streaming (`load_dataset(..., streaming=True)` + `itertools.islice(..., 1000)`), never bulk-clone.

- **Pretraining:**
  - `bigcode/starcoderdata` — ~783GB parquet, gated (accept Terms + share contact info). Load per-language: `load_dataset("bigcode/starcoderdata", data_dir="python", split="train")`. Practical choice for the 1000-row slice.
  - `bigcode/the-stack-v2` — 427GB but contains file IDs only; file contents require Software Heritage S3 credentials. Not practical here.
- **Fine-tuning (SFT distill):**
  - `WithinUsAI/claude_mythos_distilled_25k` — 25k rows, 55MB, chat `messages` format, Apache-2.0. Smallest/easiest first candidate.
- **Alignment / traces (candidates, user still picking):**
  - `aisamdasu/algocean-fable5-traces` — 30.3k train + 512 test, 786MB, tool-use trace JSONL (`prompt`/`completion`).
  - `TRACCERR/Sumtables-Cuniform-Small-Fable5-Remaster-v2` — 68.8k rows but 14.8GB and mostly vision/parquet; too big for the text-only pipeline.

## Command log (keep updated)

- `git remote add origin https://github.com/E1merM0ren0/Vulcan-AI-test.git`
- `git add .gitignore vulcan/train_out.txt && git commit -am "Add initial state"` → `b44c2a3`
- `git push -u origin main` → pushed
- `git add AGENTS.md vulcan/CMakeLists.txt vulcan/src vulcan/shaders && git commit -m "Add Vulcan GPU kernel sources, shaders, and CMake build"` → `acbf22b`, rebased onto remote `9fd7f13` → pushed as `b770f33`
- Dataset pipeline plan: shallow-slice first 1000 rows → rebuild via `tools/stream_*.py` (must be rewritten) → run `vulcan_bitnet_gpt` → log exact commands here → commit + push.
