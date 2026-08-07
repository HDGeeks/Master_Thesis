#!/bin/bash
set -e

source .venv/bin/activate

# Qwen3-8B is a full (non-GGUF) checkpoint too now, so it runs through
# the same HF script as everything else, one consistent backend for all
# sizes, the separate llama.cpp version has been retired.
#MODELS=("Qwen3.5-0.8B" "Qwen3.5-4B" "Qwen3.5-9B" "Qwen3.5-27B" "Qwen3-8B")
MODELS=("Qwen3.5-27B")
FIELDS=("abstract")

# The Qwen3.5-* sizes already have 1 run each, this adds 2 more for
# consistency (3 total). Qwen3-8B has none yet, so this is its first 2.
for model in "${MODELS[@]}"; do
  for field in "${FIELDS[@]}"; do
    for run in 1 2 3; do
      echo "=== $model / $field / extra run $run ==="
      uv run python3 src/experiment_qwen3.5_hf_1.py --model "$model" --field "$field" --quantize 4bit
    done
  done
done
