#!/usr/bin/env bash
# Train all three 3B variants (plain, bitnet158, tbn158) sequentially on
# 2× A100 80 GB via FSDP. Each run targets ~10B tokens.
#
# Each run uses --resume so a crash + rerun picks up at the latest checkpoint.
# Stdout/stderr are tee'd to logs/3B_<variant>.log for after-the-fact debugging.
#
# Track progress in WandB: `train/num_input_tokens_seen` should climb to ~10B
# (= 38,000 steps × 262K tokens/step) over each run.
#
# Usage: bash scripts/train_3b_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/accelerate_fsdp_2gpu.yaml"
VARIANTS=(plain bitnet158 tbn158)

mkdir -p logs

for VARIANT in "${VARIANTS[@]}"; do
    echo
    echo "================================================================"
    echo "  Training 3B / ${VARIANT}   ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "================================================================"
    accelerate launch --config_file "${CONFIG}" \
        pretrain.py --size 3B --variant "${VARIANT}" --resume \
        2>&1 | tee "logs/3B_${VARIANT}.log"
done

echo
echo "All 3 variants finished. Run scripts/eval.sh on each output dir:"
for VARIANT in "${VARIANTS[@]}"; do
    echo "  bash scripts/eval.sh outputs/3B_${VARIANT}/final"
done
