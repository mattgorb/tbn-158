#!/usr/bin/env bash
# Merge an FSDP SHARDED_STATE_DICT checkpoint into a single HF-loadable
# safetensors model, suitable for lm-evaluation-harness.
#
# Usage: bash scripts/consolidate_fsdp.sh <sharded_ckpt_dir>
# Example: bash scripts/consolidate_fsdp.sh outputs/3B_tbn158/final
#
# Writes the consolidated model into <sharded_ckpt_dir>/consolidated/.
# Run scripts/eval.sh against that consolidated/ dir, not the original.

set -euo pipefail

CKPT="${1:?usage: bash scripts/consolidate_fsdp.sh <sharded_ckpt_dir>}"
OUT="${CKPT}/consolidated"

accelerate merge-weights --output_path "${OUT}" --safe_serialization "${CKPT}/pytorch_model_fsdp_0"

# Copy tokenizer/config so the consolidated dir is a self-contained HF model:
for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do
    if [ -f "${CKPT}/${f}" ]; then
        cp "${CKPT}/${f}" "${OUT}/"
    fi
done

echo
echo "✓ Consolidated checkpoint at ${OUT}"
echo "  Evaluate with: bash scripts/eval.sh ${OUT}"
