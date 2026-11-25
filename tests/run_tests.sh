#!/usr/bin/env bash
# Runs the test-suite. Needs a build in ./build and a directory of test models (see README).
#   tests/run_tests.sh <models-dir> [python]
# Environment: GFX_DEVICE selects the Vulkan device (default: best available).
set -u
MODELS=${1:?usage: tests/run_tests.sh <models-dir> [python]}
PY=${2:-python3}
GFXI=$(dirname "$0")/../build/gfxi
fail=0
run() { echo "--- $*"; "$@" || { echo "FAILED: $*"; fail=1; }; }

shopt -s nullglob
ggufs=("$MODELS"/*/*.gguf)
[ ${#ggufs[@]} -gt 0 ] || { echo "no .gguf files under $MODELS/*/"; exit 2; }

# 1. Weight decoders against the gguf Python package (optional: needs `pip install gguf numpy`).
if "$PY" -c "import gguf, numpy" 2>/dev/null; then
  raw=(); for t in 10 11 16 17 18 20 21 22 23 40; do [ -f "$MODELS/samples/q_$t.bin" ] && raw+=(--raw "$t:$MODELS/samples/q_$t.bin"); done
  run "$PY" "$(dirname "$0")/test_quant.py" "$GFXI" "${raw[@]}" "${ggufs[@]}"
else echo "--- skipping decoder test (python packages gguf, numpy not installed)"; fi

# 2. Tokenizer against Hugging Face tokenizers (optional: needs `pip install tokenizers` and tokenizer.json files).
if "$PY" -c "import tokenizers" 2>/dev/null; then
  pairs=(); for d in "$MODELS"/*/; do tj="$d/tokenizer.json"; g=("$d"/*.gguf); [ -f "$tj" ] && [ ${#g[@]} -gt 0 ] && pairs+=("${g[0]}=$tj"); done
  [ ${#pairs[@]} -gt 0 ] && run "$PY" "$(dirname "$0")/test_tokenizer.py" "$GFXI" "${pairs[@]}"
else echo "--- skipping tokenizer test (python package tokenizers not installed)"; fi

# 3. Quantised matvec kernels on raw weight samples.
for spec in 23:5120 20:5120 11:5120 21:5120 18:5120 22:5120 17:5120 16:5120 40:2304; do
  t=${spec%%:*}; n=${spec##*:}
  [ -f "$MODELS/samples/q_$t.bin" ] && run "$GFXI" test-matvec "$t" "$MODELS/samples/q_$t.bin" "$n"
done

# 4. GPU engine against the CPU reference, end to end, for every model.
for g in "${ggufs[@]}"; do
  run "$GFXI" compare "$g" -n 3 --tol 0.01
done

# 5. Conversation cache (continuing, branching and returning to processed context) against plain processing.
first_of() { for g in "${ggufs[@]}"; do case "$(basename "$g")" in $1) echo "$g"; return;; esac; done; }
xml_model=$(first_of 'Qwen3.[5-9]*'); json_model=$(first_of 'Qwen3-*')
for g in "$xml_model" "$json_model"; do [ -n "$g" ] && run "$GFXI" test-context "$g"; done
[ -n "$xml_model" ] && run "$GFXI" test-context "$xml_model" --kv8

# 6. Server: prompt formatting against the models' own chat templates (optional: needs `pip install jinja2`),
#    and the reply parser.
if "$PY" -c "import jinja2" 2>/dev/null; then
  run "$PY" "$(dirname "$0")/test_chat.py" "$GFXI" "${ggufs[@]}"
else echo "--- skipping chat-format test (python package jinja2 not installed)"; fi
ink_model=$(first_of 'Inkling*')
[ -n "$xml_model" ] && run "$PY" "$(dirname "$0")/test_reply.py" "$GFXI" "$xml_model" ${json_model:+"$json_model"} ${json_model:+${ink_model:+"$ink_model"}}

# 6b. Inkling 2: the GPU engine against its CPU reference (slow; needs the whole model file).
#     GFX_MAX_BINDING_MIB lets Mesa's CPU driver take the expert buffers; a real GPU does not need it.
[ -n "$ink_model" ] && [ "$(stat -c %s "$ink_model")" -gt 1000000000 ] && run "$GFXI" compare "$ink_model" -p "<|message_user|><|content_text|>Name a prime.<|end_message|>" -n 2 --batch 8 --tol 0.02

# 7. Server end to end through the Anthropic SDK (optional: needs `pip install anthropic`).
if [ -n "$xml_model" ] && "$PY" -c "import anthropic" 2>/dev/null; then
  port=${GFX_TEST_PORT:-18181}
  "$GFXI" serve "$xml_model" --port "$port" --ctx 4096 --temp 0 2>/tmp/gfxi-serve.$$.log & server=$!
  for i in $(seq 120); do curl -s --noproxy '*' -o /dev/null "http://127.0.0.1:$port/health" && break; sleep 1; done
  NO_PROXY=127.0.0.1 no_proxy=127.0.0.1 run "$PY" "$(dirname "$0")/test_server.py" "http://127.0.0.1:$port"
  kill $server 2>/dev/null; wait $server 2>/dev/null; rm -f /tmp/gfxi-serve.$$.log
else echo "--- skipping server test (needs a Qwen 3.5+ test model and the python package anthropic)"; fi

[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail
