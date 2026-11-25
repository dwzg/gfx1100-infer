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
  raw=(); for t in 11 20 21 23 40; do [ -f "$MODELS/samples/q_$t.bin" ] && raw+=(--raw "$t:$MODELS/samples/q_$t.bin"); done
  run "$PY" "$(dirname "$0")/test_quant.py" "$GFXI" "${raw[@]}" "${ggufs[@]}"
else echo "--- skipping decoder test (python packages gguf, numpy not installed)"; fi

# 2. Tokenizer against Hugging Face tokenizers (optional: needs `pip install tokenizers` and tokenizer.json files).
if "$PY" -c "import tokenizers" 2>/dev/null; then
  pairs=(); for d in "$MODELS"/*/; do tj="$d/tokenizer.json"; g=("$d"/*.gguf); [ -f "$tj" ] && [ ${#g[@]} -gt 0 ] && pairs+=("${g[0]}=$tj"); done
  [ ${#pairs[@]} -gt 0 ] && run "$PY" "$(dirname "$0")/test_tokenizer.py" "$GFXI" "${pairs[@]}"
else echo "--- skipping tokenizer test (python package tokenizers not installed)"; fi

# 3. Quantised matvec kernels on raw weight samples.
for spec in 23:5120 20:5120 11:5120 40:2304; do
  t=${spec%%:*}; n=${spec##*:}
  [ -f "$MODELS/samples/q_$t.bin" ] && run "$GFXI" test-matvec "$t" "$MODELS/samples/q_$t.bin" "$n"
done

# 4. GPU engine against the CPU reference, end to end, for every model.
for g in "${ggufs[@]}"; do
  run "$GFXI" compare "$g" -n 3 --tol 0.01
done
[ $fail -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $fail
