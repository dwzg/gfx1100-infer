# gfx1100-infer

A from-scratch LLM inference engine for the AMD Radeon RX 7900 XTX (Navi 31, `gfx1100`),
built on Vulkan compute. No ROCm, no llama.cpp, no ML framework: it reads GGUF files directly
and runs the model with its own compute kernels.

**Status: the engine is functionally complete for token-by-token generation and verified for
correctness, but it has not yet run on a 7900 XTX.** Everything here was developed in a
container without a GPU. Nothing in this repository is a measured 7900 XTX speed.

## What works

| Area | State |
|---|---|
| Qwen 3.5 / 3.6 / 3.8 dense models (`qwen35`: gated delta-net layers + gated full attention) | Runs on the GPU path; matches the CPU reference |
| Llama-style models (`llama`, `qwen2`, `qwen3`) | Runs on the GPU path; matches the CPU reference |
| Weight formats with a native GPU layout | Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, IQ4_XS, IQ4_NL, NVFP4 |
| Other weight formats (IQ3_S, Q2_K, Q4_1, Q5_0, Q5_1, F16, F32; MXFP4 and BF16 untested) | Stored as half precision on the GPU |
| Tokenizer | Byte-pair vocabularies with the `qwen35`, `qwen2`, `llama3`, `inkling2`/`gpt-4o`, `smollm`, `gpt-2` rules |
| Attention cache | Half precision, or 8-bit with `--kv8` |
| Inkling 2 Mini (`inkling2_loop`) | **Not yet.** Its weight format and tokenizer are supported; its model graph is not |
| Batched prompt processing | **Not yet.** Prompt tokens are processed one at a time, at generation speed |
| Images, chat templates beyond ChatML, YaRN context extension, SentencePiece vocabularies | Not yet |

## Build

Needs a C++20 compiler, CMake, the Vulkan headers and loader, and `glslangValidator`.

```sh
# Debian/Ubuntu: apt install build-essential cmake ninja-build libvulkan-dev glslang-tools
cmake -S . -B build -G Ninja
cmake --build build
```

At run time it needs a Vulkan 1.3 driver. For the 7900 XTX that is Mesa's RADV; Mesa 25.2.8 is
the version checked here.

## Use

```sh
build/gfxi devices                          # which device, and which hardware features are usable
build/gfxi plan  model.gguf --vram 24       # GPU memory needed and how much context fits
build/gfxi run   model.gguf --chat -p "Why is the sky blue?" -n 200
build/gfxi run   model.gguf --kv8 --ctx 200000 -p @long_prompt.txt
build/gfxi bench                            # memory roofline, kernel throughput, dispatch cost
build/gfxi bench model.gguf                 # tokens per second
build/gfxi compare model.gguf               # GPU engine vs. CPU reference, logit by logit
build/gfxi kernels matvec_q4_k --code       # the machine code the driver generated for a kernel
```

`run` defaults to greedy decoding; add `--temp 0.7 --top-k 20 --top-p 0.8` to sample.
`--device cpu` uses the slow scalar reference path. `GFX_DEVICE=<index|name>` picks a Vulkan device.

### What fits in 24 GB

`gfxi plan` on the real `Qwen3.8-27B-UD-Q4_K_M.gguf` (computed from its tensor list, not measured):

```
weights on GPU   14.70 GiB  (native layouts 14.36, half-precision fallback 0.34)
embedding table  stays on the CPU (0.67 GiB in the file)
fixed state      0.15 GiB  (linear-attention layers, independent of context)
cache per token  64.0 KiB half precision, 34.0 KiB 8-bit
context that fits  135189 tokens (half-precision cache), 254473 tokens (8-bit cache)
  bandwidth ceiling at   4096 tokens of context:  58.6 tok/s
  bandwidth ceiling at 131072 tokens of context:  38.9 tok/s (half precision),  46.5 tok/s (8-bit)
```

The "bandwidth ceiling" is 960 GB/s divided by the bytes read per token. It is an upper bound
from the card's memory specification, not a prediction.

## Design

**Vulkan compute on RADV.** Checked against the Mesa source and by running RADV's shader
compiler in its hardware-less mode as a Navi 31:

- `VK_KHR_cooperative_matrix` exposes the card's matrix instructions (WMMA): 16x16x16,
  half precision to single precision, and 8-bit integer to 32-bit.
- Packed 8-bit integer dot products compile to `v_dot4_i32_iu8`.
- Compute shaders can be built for wave size 32 or 64.
- A storage buffer binding can be up to 4 GiB.

**Generation is limited by memory bandwidth**, so the engine is organised around reading each
weight exactly once per token, as cheaply as possible:

- *Weights keep the exact bits of their GGUF encoding* but are re-laid-out plane-major
  (`src/engine/pack.h`): with one lane per output row, the 64 lanes of a wave read 64 adjacent
  words for every load, and the input vector is the same for all lanes, so the driver loads it
  through the scalar path. In the generated code for Q4_K the inner loop is 8 coalesced weight
  loads and 4 scalar loads of 16 inputs each per 64 columns.
- *Per-block offsets stay out of the inner loop.* Every format decodes as
  `scale * (code - offset)`; the offset term only needs the sum of the input over the block,
  which is computed once per input vector.
- *One submission per token.* The whole forward pass is recorded once as a command buffer and
  resubmitted; position-dependent values come from a small host buffer. Sync points are placed
  only where one dispatch reads what an earlier one wrote.
- *The embedding table stays on the CPU* (one row is decoded per token), unless it is also the
  output projection.

**Correctness first.** `src/core/ref.cpp` is a plain scalar implementation of the forward pass.
The GPU engine is checked against it logit by logit, and it is itself checked by generating
coherent text on real models.

## Verification

What was checked, and how:

| Check | Against | Result |
|---|---|---|
| Weight decoders, 14 quantised formats (all but MXFP4) plus F16 and F32 | `gguf` Python package | Bit-identical |
| Tokenizer, 5 model families, 442 strings each | Hugging Face `tokenizers` | Identical |
| Quantised matvec kernels on weights sampled from the real Qwen 3.8 and Inkling files | Scalar decoder | Relative error about 1e-7 |
| GPU engine end to end, 12 model files across 4 architectures (every native format except IQ4_XS and NVFP4, which have kernel tests only) | CPU reference | Relative logit error under 2.5e-3, same top token |
| 8-bit cache | CPU reference | Relative logit error 0.4-0.9% on Qwen 3.5 0.8B; 4.4% on a 135M Llama-style model |
| Every kernel compiles for the 7900 XTX | RADV 25.2.8 as Navi 31 (no hardware) | All compile, no register spills |

Not checked:

- **Anything on a real 7900 XTX.** The GPU path ran only on Mesa's CPU Vulkan driver (lavapipe).
  That exercises the same SPIR-V with a wave size of 8; the real card uses 32 or 64. No kernel
  depends on the wave size for correctness, but that has not been observed on hardware.
- **Agreement with llama.cpp.** Building it as a cross-check was not possible in the development
  session. The decoders and tokenizer were checked against independent implementations; the
  forward pass was checked only by text quality.
- **Qwen 3.8 27B itself.** The 16 GB file was not downloaded; its architecture was tested with
  Qwen 3.5 0.8B, which has the same graph, and its weight formats with samples from the real file.

Run the suite with `tests/run_tests.sh <models-dir> [python]`.

## Next, on the workstation

In order:

1. `gfxi devices`, then `tests/run_tests.sh` on the real card.
2. `gfxi bench` and `gfxi bench model.gguf`. This gives the achievable memory bandwidth, each
   kernel as a fraction of it, and the cost of a dispatch plus sync point.
3. Tune from those numbers: wave size, loads in flight per wave, splitting columns across more
   waves, folding the block-sum pass into the kernels that produce the vector, and a faster
   codebook lookup for IQ4 and NVFP4 (those kernels currently cost about 3x the arithmetic of
   Q4_K per weight).
4. Batched prompt processing using the matrix instructions. Without it a 100K-token prompt takes
   as long as generating 100K tokens.
5. The Inkling 2 Mini architecture.
6. Saving and restoring a processed context, so a long prompt is only paid for once.

## Layout

```
src/core/     GGUF reader, weight decoders, tokenizer, model description, CPU reference, sampler
src/vk/       Vulkan device, memory, pipelines, recorded programs
src/engine/   GPU weight layouts and the GPU engine
src/shaders/  compute kernels (GLSL)
src/tools/    gfxi command line
tests/        decoder, tokenizer and end-to-end tests
```

`src/core/iq3s_grid.inc` is the IQ3_S codebook, a constant of that GGUF format, with values taken
from ggml (MIT licence).
