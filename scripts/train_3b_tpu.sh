#!/usr/bin/env bash
# Train the 3B variants on TPU v6e. Default: single-host v6e-8 + all 3 variants.
#
# Usage:
#   bash scripts/train_3b_tpu.sh                       # all 3 variants on v6e-8
#   bash scripts/train_3b_tpu.sh tbn158                # one variant on v6e-8
#   ACCEL=configs/accelerate_tpu_v6e_32.yaml \
#       bash scripts/train_3b_tpu.sh                   # use multi-host v6e-32 config
#
# Multi-host note: this script runs `accelerate launch` on whichever host you
# SSH'd into. For v6e-32 (4 hosts), invoke from your laptop instead so the
# command fans out to every worker:
#   gcloud compute tpus tpu-vm ssh <tpu-name> --zone=us-east1-d --worker=all \
#       --command='cd ~/tbn-158 && ACCEL=configs/accelerate_tpu_v6e_32.yaml \
#                    bash scripts/train_3b_tpu.sh tbn158'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ACCEL="${ACCEL:-configs/accelerate_tpu_v6e_8.yaml}"

if [ $# -eq 0 ]; then
    VARIANTS=(plain bitnet158 tbn158)
else
    VARIANTS=("$@")
fi

mkdir -p logs

for VARIANT in "${VARIANTS[@]}"; do
    echo
    echo "================================================================"
    echo "  TPU train: 3B / ${VARIANT}   ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo "  Accelerate config: ${ACCEL}"
    echo "================================================================"
    accelerate launch --config_file "${ACCEL}" \
        pretrain.py --size 3B --variant "${VARIANT}" \
        --config "configs/3B_${VARIANT}_tpu.yaml" \
        --resume \
        2>&1 | tee "logs/3B_${VARIANT}_tpu.log"
done

echo
echo "Done. Consolidate + eval each final checkpoint:"
for V in "${VARIANTS[@]}"; do
    echo "  bash scripts/consolidate_fsdp.sh outputs/3B_${V}/final"
    echo "  bash scripts/eval.sh             outputs/3B_${V}/final/consolidated"
done
