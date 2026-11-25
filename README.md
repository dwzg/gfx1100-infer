# gfx1100-infer

A from-scratch LLM inference engine for the AMD Radeon RX 7900 XTX (Navi 31, `gfx1100`),
built on Vulkan compute. No ROCm, no llama.cpp, no ML framework: it reads GGUF files directly
and runs the model with its own compute kernels.

It includes a server that speaks the Anthropic Messages API, so
[Claude Code](#claude-code) can use a local model as its backend.

**Status: the engine and the server are functionally complete and verified for correctness, but
nothing has run on a 7900 XTX yet.** Everything here was developed in a container without a GPU.
Nothing in this repository is a measured 7900 XTX speed.

## What works

| Area | State |
|---|---|
| Qwen 3.5 / 3.6 / 3.8 dense models (`qwen35`: gated delta-net layers + gated full attention) | Runs on the GPU path; matches the CPU reference |
| Llama-style models (`llama`, `qwen2`, `qwen3`) | Runs on the GPU path; matches the CPU reference |
| Weight formats with a native GPU layout | Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, NVFP4 |
| Other weight formats (Q2_K, Q4_1, Q5_0, Q5_1, F16, F32; MXFP4 and BF16 untested) | Stored as half precision on the GPU |
| Inkling's expert matrices | NVFP4 and Q5_K (what its `UD-Q4_K_M` file uses) and the other formats without a codebook table; all 128 experts of a layer share one buffer |
| Tokenizer | Byte-pair vocabularies with the `qwen35`, `qwen2`, `llama3`, `inkling2`/`gpt-4o`, `smollm`, `gpt-2` rules |
| Attention cache | Half precision, or 8-bit with `--kv8`; context sized automatically with `--ctx max` |
| Saving and restoring a processed context | `--save-state` / `--load-state` |
| Inkling 2 Mini (`inkling2_loop`) | Runs on the GPU path and matches its CPU reference; served to Claude Code in its own message format. Tested on the CPU driver with only the experts a test needs uploaded (the whole model needs 15 GiB of device memory) |
| Batched prompt processing | Yes: prompt tokens go through in batches (default 32) with weights decoded once per 4-token tile |
| Matrix instructions (WMMA) for prompts | Written and compiled for the card, **never executed**; enabled per weight format only if an automatic on-device check passes |
| Interactive chat | `gfxi chat`, ChatML-style templates only, with `--system` and `--no-think` |
| Messages API server (`gfxi serve`) | Streaming, tool calls, reasoning blocks, token counting; drives the real Claude Code CLI end to end (on a CPU driver, with a small model) |
| Reuse of processed context between requests | Several conversations share the context memory; a request only pays for what no earlier one has processed |
| Images, chat templates other than ChatML, YaRN context extension, SentencePiece vocabularies | Not yet |

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
build/gfxi chat  model.gguf --kv8 --ctx max                 # interactive, context kept between turns (Qwen-style and Inkling models)
build/gfxi serve model.gguf --kv8 --ctx max                 # Messages API server, e.g. for Claude Code
build/gfxi run   model.gguf --kv8 --ctx max -p @long_prompt.txt --save-state doc.ctx   # process once
build/gfxi run   model.gguf --kv8 --load-state doc.ctx --chat -p "Summarise it."         # reuse
build/gfxi bench                            # memory roofline, kernel throughput, dispatch cost
build/gfxi bench model.gguf                 # prompt and generation tokens per second
build/gfxi compare model.gguf               # GPU engine vs. CPU reference, logit by logit
build/gfxi kernels matvec_q4_k --code       # the machine code the driver generated for a kernel
```

`run` defaults to greedy decoding; add `--temp 0.7 --top-k 20 --top-p 0.8` to sample.
`--device cpu` uses the slow scalar reference path. `GFX_DEVICE=<index|name>` picks a Vulkan device.

### Claude Code

`gfxi serve` makes the model available over the Anthropic Messages API, which is what Claude Code
speaks. Start the server, then start Claude Code with the settings the server prints:

```sh
build/gfxi serve Qwen3.8-27B-UD-Q4_K_S.gguf --kv8 --prompt-cache ~/.cache/gfxi   # listens on 127.0.0.1:8080

export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=local                       # any text, unless the server has --api-key
export ANTHROPIC_MODEL=Qwen3.8-27B-UD-Q4_K_S            # the name the server reports (--model-id)
export ANTHROPIC_DEFAULT_HAIKU_MODEL=$ANTHROPIC_MODEL   # background requests (titles, summaries)
export CLAUDE_CODE_SUBAGENT_MODEL=$ANTHROPIC_MODEL
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=229376            # printed by the server for its context size
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
claude
```

The same variables can go in the `env` block of `~/.claude/settings.json`. Claude Code does not
know the model, so without `CLAUDE_CODE_MAX_CONTEXT_TOKENS` it assumes a window of a million
tokens; with it, it compacts the conversation before the server's context is full. Anthropic does
not support Claude Code with models other than Claude: expect it to work as well as the model
follows a long system prompt and calls tools reliably, which is the model's doing, not the server's.

What the server does with a request:

- **Formats it as the model was trained.** The system prompt, tool definitions, tool calls, tool
  results and reasoning are written in the model's own chat format (read from the template in the
  model file; ChatML with tool calls as `<function=...>` blocks, as Qwen 3.5 and later use, or as
  JSON, as Qwen 3 and Hermes-style models use). The reply is split back into `thinking`, `text`
  and `tool_use` blocks and streamed as it is generated. Text that comes from outside (file
  contents, tool results, the user's messages) is never read as markup: a file containing
  `<|im_end|>` cannot end a message.
- **Processes each token of a conversation once.** Claude Code resends the whole conversation with
  every request. The server keeps what it has processed on the GPU and only runs the new part: in
  the test run below, the second request of a tool loop reused 1,570 of 1,599 prompt tokens.
  `usage` reports reused tokens as `cache_read_input_tokens`.
- **Keeps several conversations at once.** The main conversation, background requests (titles,
  summaries) and subagents each keep their place; the context memory is handed out to them in
  pages of 1,024 tokens, and conversations with the same beginning (two subagents with the same
  instructions, a conversation and a side question about it) share the pages of that beginning.
  When the memory runs out, conversations idle for an hour go first, then the cheapest to redo.
- **Remembers the system prompt across restarts.** With `--prompt-cache DIR`, the processed
  context at the end of the tool list and at the end of the system prompt is written to a file
  the first time it is computed (if it took at least 512 tokens). A later request that begins the
  same way, after a restart or in a new session, starts from the file instead of processing
  Claude Code's whole prompt again. The eight most recently used files are kept; for Qwen 3.8 27B
  with the 8-bit cache a file is roughly 35 KiB per token plus 150 MiB.
- **Keeps reasoning in the conversation.** Earlier reasoning stays in the prompt (as Qwen 3.8's
  template does by default), so the conversation only ever grows at its end. `--drop-thinking`
  follows templates that drop it instead, at the cost of reprocessing each turn.

Options: `--think auto|on|off` (default: reason when the request enables thinking, which Claude
Code does), `--effort low|medium|high` (overrides the request; Qwen 3.8's template has effort
levels), `--temp/--top-k/--top-p` (defaults 0.6 / 20 / 0.95; the request's own sampling settings
are ignored unless `--client-sampling`), `--host`, `--port`, `--api-key`, `--model-id`,
`--conversations N`, `--checkpoints N`, `--prompt-cache DIR`, `--log-prompts`, `--dump-requests DIR`.

Limits to know about:

- One request generates at a time; others wait their turn (they are kept alive with pings).
- Claude Code gives up on a request that has produced nothing after ten minutes (`API_TIMEOUT_MS`)
  and sends it again. The server keeps what it had processed, so the retry continues from there
  (seen in the test run on the CPU driver, where the first prompt took twenty minutes).
- The context is shared by all conversations the server keeps, which is why the window given to
  Claude Code above is smaller than the server's context.
- Qwen 3.5 and later mix attention with linear-attention layers, whose state cannot be rewound.
  The server keeps copies of that state at useful points (the end of the tool list, the end of the
  system prompt, the end of each request); a request that diverges from a conversation elsewhere
  reprocesses from the nearest earlier copy. For Qwen 3.8 27B a copy is about 150 MiB of main
  memory, and up to 8 conversations x 5 copies are kept (`--conversations`, `--checkpoints`).
- Not implemented: images and documents (replaced by a placeholder), structured-output schemas
  (not enforced), server-side tools such as web search (ignored), and `tool_choice` beyond forcing
  a call.

### Getting the most context out of 24 GB

Three things decide how much context fits: how much memory the weights take, how many bytes each
token of context costs, and how much the desktop is already using.

- `--kv8` stores the attention cache in 8 bits instead of 16: 34.1 KiB per token instead of 64.1 KiB
  for Qwen 3.8 27B. It costs some accuracy (see Verification).
- `--ctx max` sizes the context from the GPU memory that is actually free after the weights are
  loaded, minus `--reserve` (default 0.25 GiB). Close browsers and other GPU-heavy programs first.
- Pick the quantisation with `gfxi plan`. For Qwen 3.8 27B, with 1.5 GiB assumed for the desktop
  (computed from the files' tensor lists, not measured):

| File | Weights on GPU | Context, half-precision cache | Context, 8-bit cache |
|---|---|---|---|
| `UD-Q4_K_M` | 14.44 GiB | 126,918 | 238,521 |
| `UD-Q4_K_S` | 13.49 GiB | 142,439 | 262,144 (the model's trained limit) |
| `UD-IQ4_XS` | 12.66 GiB | 156,022 | 262,144 |
| `UD-Q3_K_XL` | 11.79 GiB | 170,290 | 262,144 |

So `UD-Q4_K_S` with `--kv8` reaches the model's full 262,144 tokens on a card that also drives a
desktop. Going past 262,144 would need YaRN position scaling, which is not implemented.

Inkling 2 Mini (`UD-Q4_K_M`, 15.0 GiB of weights on the GPU) keeps a context-length cache in only
8 of its 48 cache slots; the other 40 keep a 1,024-token sliding window. Its cache therefore costs
32.0 KiB per token at half precision and 17.0 KiB at 8 bits, and with the same 1.5 GiB left for
the desktop `gfxi plan` gives **228,607 tokens (half precision) or 435,120 tokens (8-bit)**.

Two things to know before relying on a very long context:

- **Filling it takes a while.** Prompts are processed in batches, but how fast that is on the real
  card is unmeasured, and the linear-attention layers still walk the tokens in order. Save a long
  prompt once with `--save-state` and reuse it.
- **Generation slows down as the context fills**, because the whole cache is read for every token.
  Upper bounds from the card's 960 GB/s memory bandwidth for `UD-Q4_K_S` (not predictions):
  64 tok/s at 4K tokens, 49 tok/s at 131K and 40 tok/s at 262K with the 8-bit cache.

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
- *One submission per token (or per prompt batch).* The forward pass is recorded once per batch
  size as a command buffer and resubmitted; position-dependent values come from small host
  buffers. Sync points are placed only where one dispatch reads what an earlier one wrote.
- *Prompts in batches.* Every kernel takes a token dimension. The matrix kernels are written
  against a few accumulator operations that are scalar for generation, 4 tokens wide for prompts
  (each weight decoded once per tile), or emit decoded weights into shared memory for the
  matrix-instruction path, which multiplies 16x16 tiles for 16 tokens at a time.
- *The embedding table stays on the CPU* (one row is decoded per token), unless it is also the
  output projection.
- *The attention cache is read once per token, in parallel.* Keys are stored plane-major like the
  weights, so scoring is the same kernel shape with one lane per cached position, and all query
  heads that share a key head are scored in one pass. Value sums run over 1024-position segments
  in parallel. Dispatch sizes follow the filled context through indirect dispatch.
- *The attention cache is paged.* It is divided into pages of 1,024 positions, and the three
  kernels that touch it go through a page table (one extra scalar load per workgroup). A token
  stream owns a list of pages, so several conversations share the memory, a conversation can
  branch off another by sharing its pages (a shared page is copied when written to), and nothing
  is ever moved. The linear-attention state lives outside the cache and is copied to main memory
  where a conversation may later be resumed (`src/engine/context.h`).

**Inkling 2** (`src/core/inkling.h` describes the architecture) adds, on top of the above:

- *Experts picked on the GPU.* A layer's 128 experts sit in one buffer, each in the plane-major
  layout. A small kernel routes each token (sigmoid gate, top 8), and the matrix kernels get an
  "expert" mode that reads the chosen expert's index and offsets every weight load by it, so a
  token touches only the 8 experts it uses and nothing is copied or selected on the CPU.
- *Two kinds of cache.* Layers that attend to everything use the paged cache; the others use a
  ring of 1,024 positions plus one batch, addressed by position modulo its size. The rings and the
  convolution history are small and travel with a conversation like the linear-attention state.
- *Position bias and sinks in the score kernel.* A table of bias by distance is computed per query
  head once per token; the score kernel adds its entry for each position's distance, and the
  softmax kernel adds the sink's term.
- *The loop is unrolled when the forward pass is recorded*: 48 slot executions. Consecutive
  steps are fused into single kernels where one would otherwise wait for the other (mixing with
  the norm that follows it, the message controllers, routing with its matrix product, the bias
  table with the cache write, the experts' sum with the convolution and the residual), which
  leaves 1,292 dispatches and 896 synchronisation points per token (down from 1,963 and 1,449).

**Correctness first.** `src/core/ref.cpp` is a plain scalar implementation of the forward pass.
The GPU engine is checked against it logit by logit, and it is itself checked by generating
coherent text on real models.

## Verification

What was checked, and how:

| Check | Against | Result |
|---|---|---|
| Weight decoders, 18 quantised formats (all but MXFP4) plus F16 and F32 | `gguf` Python package | Bit-identical |
| Tokenizer, 5 model families, 442 strings each | Hugging Face `tokenizers` | Identical |
| Quantised matvec kernels (IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, Q3_K, NVFP4) on weights sampled from the real Qwen 3.8 and Inkling files | Scalar decoder | Relative error about 1e-7 |
| GPU engine end to end, 12 model files across 4 architectures | CPU reference | Relative logit error under 2.5e-3, same top token |
| Long context: a 1,251-token prompt, and a start at position 120,000 over an empty cache | CPU reference | Same top token; relative error 7e-4 (half precision), 6e-3 to 2e-2 (8-bit) |
| Restored context | Uninterrupted run | Bit-identical logits |
| Batched prompts (batch sizes 1, 5, 7, 32; forced query sub-batches), all 12 model files | CPU reference | Same results as token-by-token |
| Batched and emulated matrix-path kernels, all 14 native formats, on real weight samples | Scalar decoder | Relative error about 3e-7 (batched) and 3e-4 (half-precision matrix path) |
| 8-bit cache | CPU reference | Relative logit error 0.4-0.9% on Qwen 3.5 0.8B; 1-2% on a 135M Llama-style model |
| Every kernel compiles for the 7900 XTX, including the matrix-instruction and Inkling ones | RADV 25.2.8 as Navi 31 (no hardware) | All 107 compile, none spills vector registers; the batched kernels of seven formats spill 27-38 scalar registers (to tune on the card); the matrix kernels contain `v_wmma_f32_16x16x16_f16` |
| Paged cache with small pages handed out in reverse order, half-precision and 8-bit | CPU reference | Same errors as unpaged |
| Conversation cache: continuing, branching, returning, same prompt again, eviction (Qwen 3.5 with both cache types, Qwen 3) | Processing the same prompt from scratch | Bit-identical logits |
| Prompt formatting: 86 prompts with system text, tools, calls, results, reasoning and effort levels, on the templates of Qwen 3.8, 3.5, 3, 2.5 and SmolLM2 | The model's own template, run with jinja2 | Byte-identical |
| Reply parser: 14 replies, each fed in pieces of 1, 2, 3, 7, 64 bytes and whole | Expected blocks; events add up to the blocks; parsed and written back gives the original text | All pass |
| Server protocol: streaming, forced tool call, tool result, reuse across turns and across a side request, cancel, stop sequence, errors | Official `anthropic` Python SDK 2.14 as the client | All pass; a follow-up turn reused 432 of 432 earlier tokens |
| Inkling 2 Mini, CPU reference | The text it writes | Coherent reasoning and a correct answer ("Capital of France is Paris ... `s[::-1]`") |
| Inkling 2 Mini, GPU engine: token by token, prompts in batches of 8, 8-bit cache, and a start at position 9,000 (ring wrap-around, distances past the bias table, position scaling) | Its CPU reference | Relative logit error 7e-5 (half precision), 4e-4 at position 9,000, about 2e-2 (8-bit); same top token throughout |
| Conversation cache with the Inkling engine: continuing, a side request branching off, returning, a second conversation with the same beginning | Processing the same tokens from scratch | Bit-identical logits |
| Inkling prompt formatting (32 prompts) and reply parsing (6 cases at every piece size) | Its own template run with jinja2; expected blocks | Byte-identical; all pass |
| Prompt store: a request, a server restart, the same request and a different one | The first run | The restart begins from the stored prefix (415 of 430 tokens), gives the same answer, and the next question reuses it too |
| Claude Code itself: CLI 2.1.332 with its default system prompt and 13 tools (12,453 prompt tokens), asked to read a file and report a value from it; Qwen 3.5 0.8B on the CPU driver | The right answer | Called its Read tool, then answered correctly; the second request reused 12,558 of 12,587 prompt tokens |

Not checked:

- **Anything on a real 7900 XTX.** The GPU path ran only on Mesa's CPU Vulkan driver (lavapipe).
  That exercises the same SPIR-V with a wave size of 8; the real card uses 32 or 64. No kernel
  depends on the wave size for correctness, but that has not been observed on hardware.
- **The matrix-instruction kernels have never run.** The CPU driver available here lacks the
  extension (Mesa 26.2 adds it; building a newer one was not possible in the development session).
  Everything around the three matrix calls is tested through an emulated variant; the calls
  themselves are not. The engine checks each format on the device before using it, and
  `gfxi test-matvec` tests them directly.
- **Agreement with llama.cpp.** Building it as a cross-check was not possible in the development
  session. The decoders and tokenizer were checked against independent implementations; the
  forward pass was checked only by text quality.
- **Qwen 3.8 27B itself.** The 16 GB file was not downloaded; its architecture was tested with
  Qwen 3.5 0.8B, which has the same graph, and its weight formats with samples from the real files.
  In particular the 8-bit cache's effect on the 27B model's answers at long context is unknown.
- **A genuinely full long context.** Position 120,000 was tested over an empty cache, which checks
  the indexing but not a real 100K-token document.
- **IQ4_XS and the IQ2/IQ3 formats end to end.** No small test model uses them; they have
  decoder and kernel tests only. (NVFP4 and Q5_K are now covered end to end by Inkling.)
- **Inkling 2 Mini as a whole on one device.** The development machine has 15 GiB of memory and
  the model needs 15 GiB on the device, so the GPU tests uploaded only the experts each test used
  (the CPU reference says which) and were short: at most 19 tokens. Long prompts, the server with
  Inkling, and Claude Code with Inkling have not run. Its correctness rests on the CPU reference,
  which was checked only by the text it writes, not against another implementation.

Run the suite with `tests/run_tests.sh <models-dir> [python]`.

## Next, on the workstation

In order:

1. `gfxi devices`, then `tests/run_tests.sh` on the real card. The `test-matvec` lines labelled
   "matrix instructions" and the start-up line "matrix instructions verified for: ..." say whether
   the WMMA kernels are correct there.
2. `gfxi bench` and `gfxi bench model.gguf` (also with `--no-wmma`). This gives the achievable
   memory bandwidth, each kernel as a fraction of it, the cost of a dispatch plus sync point, and
   prompt and generation speed.
3. Tune from those numbers: wave size, loads in flight per wave, splitting columns across more
   waves, tokens per tile, a faster codebook lookup for IQ4 and NVFP4 (those kernels currently do
   about 3x the arithmetic of Q4_K per weight), and folding the half-precision conversion into the
   kernels that produce each vector.
4. Claude Code against the real model: `gfxi serve` with Qwen 3.8 27B, then a session with the
   settings it prints. Watch the server's log lines (tokens reused, processed, tokens per second)
   and how well the model handles Claude Code's tool calls.
5. Inkling 2 Mini: `gfxi compare Inkling-2-Mini-UD-Q4_K_M.gguf` (GPU against the CPU reference,
   this time with every expert on the device), then `gfxi bench` and `gfxi serve` with it. Its
   forward pass is about 1,300 small dispatches with 900 synchronisation points per token, so
   expect those, rather than memory bandwidth, to set its speed; `gfxi bench` shows what one costs.

## Layout

```
src/core/     GGUF reader, weight decoders, tokenizer, model descriptions, CPU references (ref.cpp, inkling.cpp), sampler
src/vk/       Vulkan device, memory, pipelines, recorded programs
src/engine/   GPU weight layouts, the GPU engine, the conversation cache
src/shaders/  compute kernels (GLSL)
src/server/   Messages API server: prompt formatting, reply parsing, HTTP
src/tools/    gfxi command line
tests/        decoder, tokenizer, chat-format, server and end-to-end tests
```

`src/core/iq3s_grid.inc` is the IQ3_S codebook, a constant of that GGUF format, with values taken
from ggml (MIT licence).
