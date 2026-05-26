#!/usr/bin/env bash
# Post-hoc zero-shot eval via lm-evaluation-harness on a saved checkpoint.
# Tasks are the BitNet b1.58 paper's set: ARC-easy/challenge, HellaSwag,
# WinoGrande, PIQA, BoolQ, OpenBookQA, LAMBADA.
#
# Usage: bash scripts/eval.sh <checkpoint_dir>
# Example: bash scripts/eval.sh outputs/3B_tbn158/final
#
# Requires: pip install lm-eval
#
# Results land in <checkpoint_dir>/eval_results/.

set -euo pipefail

CKPT="${1:?usage: bash scripts/eval.sh <checkpoint_dir>}"
TASKS="hellaswag,winogrande,arc_easy,arc_challenge,piqa,boolq,openbookqa,lambada_openai"
OUT="${CKPT}/eval_results"

mkdir -p "${OUT}"
lm_eval --model hf \
    --model_args "pretrained=${CKPT},dtype=bfloat16" \
    --tasks "${TASKS}" \
    --device cuda \
    --batch_size 8 \
    --output_path "${OUT}"
