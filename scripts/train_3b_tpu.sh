#!/usr/bin/env bash
# Train the 3B variants on TPU v6e. Default: single-host v6e-8 + all 3 variants.
#
# Usage:
#   bash scripts/train_3b_tpu.sh                       # all 3 variants on v6e-8
#   bash scripts/train_3b_tpu.sh tbn158                # one variant on v6e-8
#   ACCEL=configs/accelerate_tpu_v6e_32.yaml \
#       bash scripts/train_3b_tpu.sh                   # use multi-host v6e-32 config
#
# GCS-backed checkpoints (survive spot preemption — recommended):
#   GCS_BUCKET=matt-tbn158-ckpts \
#       bash scripts/train_3b_tpu.sh                   # writes outputs/ to gs://<bucket>
#
# Multi-host note: this script runs `accelerate launch` on whichever host you
# SSH'd into. For v6e-16/-32/-64, invoke from your laptop instead so the
# command fans out to every worker:
#   gcloud compute tpus tpu-vm ssh <tpu-name> --zone=<zone> --worker=all \
#       --command='cd ~/tbn-158 && ACCEL=configs/accelerate_tpu_v6e_32.yaml \
#                    GCS_BUCKET=matt-tbn158-ckpts bash scripts/train_3b_tpu.sh tbn158'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ACCEL="${ACCEL:-configs/accelerate_tpu_v6e_8.yaml}"
GCS_BUCKET="${GCS_BUCKET:-}"
# Optional suffix appended to output dir + WandB run name, e.g. RUN_SUFFIX=_runA
# → output_dir=outputs/3B_tbn158_runA, run_name=3B_tbn158_runA. Lets you run
# multiple parallel variants without colliding.
RUN_SUFFIX="${RUN_SUFFIX:-}"

# Mount GCS bucket if requested. Checkpoints land in the mounted dir and survive
# VM preemption — a fresh VM remounts the same bucket and --resume picks up at
# the latest checkpoint.
if [ -n "${GCS_BUCKET}" ]; then
    MOUNT_DIR="${HOME}/gcs"
    mkdir -p "${MOUNT_DIR}"
    if ! mountpoint -q "${MOUNT_DIR}"; then
        echo "==> Mounting gs://${GCS_BUCKET} at ${MOUNT_DIR}"
        # --implicit-dirs: treat object prefixes as directories
        # --file-mode=664 --dir-mode=775: writable by user/group (needed for HF Trainer)
        gcsfuse --implicit-dirs --file-mode=664 --dir-mode=775 \
            "${GCS_BUCKET}" "${MOUNT_DIR}"
    fi
    OUTPUT_BASE="${MOUNT_DIR}/outputs"
    echo "==> Checkpoints will persist to gs://${GCS_BUCKET}/outputs/"
else
    OUTPUT_BASE="outputs"
    echo "==> Checkpoints will go to local outputs/ (LOST on preemption — set GCS_BUCKET to persist)"
fi

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
    echo "  Output dir: ${OUTPUT_BASE}/3B_${VARIANT}${RUN_SUFFIX}"
    echo "================================================================"
    CFG="${CONFIG_FILE:-configs/3B_${VARIANT}_tpu.yaml}"

    # Pre-cache the latest checkpoint's model file so the 8 accelerate workers'
    # torch.load calls hit the kernel page cache instead of doing 8-way
    # contended gcsfuse reads. One sequential reader pulls at full
    # ~100-300 MB/s; 8 contending readers get ~5 MB/s each and look hung for
    # 30+ minutes. This single cat warms the cache in ~2 min, after which the
    # actual training load completes in seconds.
    OUT_DIR="${OUTPUT_BASE}/3B_${VARIANT}${RUN_SUFFIX}"
    LATEST_CKPT=$(ls -d "${OUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -n "${LATEST_CKPT}" ] && [ -f "${LATEST_CKPT}/pytorch_model.bin" ]; then
        echo "==> Pre-caching ${LATEST_CKPT}/pytorch_model.bin (single-stream cat)"
        time cat "${LATEST_CKPT}/pytorch_model.bin" > /dev/null
        echo "    cache warmed."
    fi

    # PYTHONUNBUFFERED=1 forces Python's stdout/stderr to be line-buffered even
    # when piped through tee. Without this, `print()` output sits in an 8 KB
    # buffer and looks like a hang when in fact training is progressing.
    PYTHONUNBUFFERED=1 accelerate launch --config_file "${ACCEL}" \
        pretrain.py --size 3B --variant "${VARIANT}" \
        --config "${CFG}" \
        --output_dir "${OUTPUT_BASE}/3B_${VARIANT}${RUN_SUFFIX}" \
        --run_name "3B_${VARIANT}${RUN_SUFFIX}" \
        --resume \
        2>&1 | tee "logs/3B_${VARIANT}${RUN_SUFFIX}_tpu.log"
done

echo
echo "Done. Consolidate + eval each final checkpoint:"
for V in "${VARIANTS[@]}"; do
    echo "  bash scripts/consolidate_fsdp.sh ${OUTPUT_BASE}/3B_${V}${RUN_SUFFIX}/final"
    echo "  bash scripts/eval.sh             ${OUTPUT_BASE}/3B_${V}${RUN_SUFFIX}/final/consolidated"
done
