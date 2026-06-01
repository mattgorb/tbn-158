#!/usr/bin/env bash
# Cron-friendly TPU watchdog. Runs periodically (e.g. every 30 min), ensures
# the TPU exists and training is alive. If the TPU is preempted, recreates it,
# runs setup on every worker, and launches training in detached tmux sessions
# on every worker (which resume from GCS via --resume).
#
# Handles BOTH single-host (v6e-8, v4-8...) and multi-host (v6e-16/32/64,
# v4-16/32) slices — `--worker=all` is used everywhere.

set -uo pipefail

# ============ CONFIG ============
TPU_NAME="${TPU_NAME:-matt-tbn158-v6e32}"
ZONE="${ZONE:-europe-west4-a}"
ACCEL_TYPE="${ACCEL_TYPE:-v6e-32}"
VERSION="${VERSION:-v2-alpha-tpuv6e}"
NETWORK="${NETWORK:-tbn158-net}"
REPO_URL="${REPO_URL:?REPO_URL env var required, e.g. https://<pat>@github.com/<you>/tbn-158.git}"
GCS_BUCKET="${GCS_BUCKET:-matt-tbn158-ckpts}"
VARIANT="${VARIANT:-bitnet158}"
USE_SPOT="${USE_SPOT:-true}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_tpu_v6e_32.yaml}"
# Passed through to train_3b_tpu.sh — used by the launch tmux command below.
RUN_SUFFIX="${RUN_SUFFIX:-}"
CONFIG_FILE="${CONFIG_FILE:-}"

HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required}"
WANDB_TOKEN="${WANDB_TOKEN:?WANDB_TOKEN env var required}"

LOG_FILE="${LOG_FILE:-/tmp/tpu_monitor.log}"

# ============ SSH KEY (cron-safe) ============
# gcloud needs an SSH key for the TPU VM. Without one, it interactively prompts
# for a passphrase, which deadlocks any non-interactive caller (cron, scripts).
# Generate an empty-passphrase key the first time we run.
if [ ! -f "$HOME/.ssh/google_compute_engine" ]; then
    mkdir -p "$HOME/.ssh"
    ssh-keygen -t rsa -f "$HOME/.ssh/google_compute_engine" -N "" -q
fi

# ============ HELPERS ============
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG_FILE"; }

get_state() {
    gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" \
        --format='value(state)' 2>/dev/null || echo "NOT_FOUND"
}

# Returns 0 if training is running on worker 0 (proxy for the whole slice).
training_is_running() {
    local count
    count=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=0 \
        --command='pgrep -fc pretrain.py || echo 0' 2>/dev/null | tr -d '[:space:]')
    [ "${count:-0}" -gt 0 ]
}

create_tpu() {
    local spot_flag=""
    [ "$USE_SPOT" = "true" ] && spot_flag="--spot"
    log "Creating TPU $TPU_NAME ($ACCEL_TYPE, zone=$ZONE, spot=$USE_SPOT)..."
    gcloud compute tpus tpu-vm create "$TPU_NAME" --zone="$ZONE" \
        --accelerator-type="$ACCEL_TYPE" --version="$VERSION" \
        --network="$NETWORK" $spot_flag 2>&1 | tee -a "$LOG_FILE"
}

wait_for_ready() {
    log "Waiting for TPU to become READY..."
    for i in $(seq 1 60); do
        local s
        s=$(get_state)
        if [ "$s" = "READY" ]; then
            log "TPU is READY."
            return 0
        fi
        sleep 30
    done
    log "TPU did not become READY within 30 min. Aborting this cycle."
    return 1
}

# Runs on every worker: clone if missing, install if missing, auth, then
# (re)launch training in a detached tmux session. Idempotent.
provision_and_launch_all_workers() {
    log "Provisioning + launching across all workers of $TPU_NAME..."
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all \
        --command="
            set -e
            if [ ! -d \$HOME/tbn-158 ]; then
                git clone $REPO_URL \$HOME/tbn-158
                bash \$HOME/tbn-158/scripts/setup_tpu_vm.sh
            fi
            grep -q '.local/bin' \$HOME/.bashrc || echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> \$HOME/.bashrc
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            huggingface-cli login --token $HF_TOKEN 2>/dev/null || true
            wandb login $WANDB_TOKEN 2>/dev/null || true
            sudo apt-get install -y tmux 2>/dev/null || true
            cd \$HOME/tbn-158 && git pull
            tmux kill-session -t train 2>/dev/null || true
            tmux new -d -s train \"ACCEL=$ACCEL_CONFIG GCS_BUCKET=$GCS_BUCKET RUN_SUFFIX=$RUN_SUFFIX CONFIG_FILE=$CONFIG_FILE bash scripts/train_3b_tpu.sh $VARIANT 2>&1 | tee /tmp/train.log\"
        " 2>&1 | tee -a "$LOG_FILE"
}

# ============ MAIN ============
log "--- tick ---"
state=$(get_state)
log "TPU state: $state"

case "$state" in
    READY)
        if training_is_running; then
            log "Training is running on worker 0. Nothing to do."
        else
            log "TPU up but no training process found. Relaunching..."
            provision_and_launch_all_workers
        fi
        ;;
    PREEMPTED|STOPPED|TERMINATED|NOT_FOUND)
        log "TPU is $state. Cleaning up + recreating..."
        gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet 2>/dev/null || true
        if create_tpu; then
            wait_for_ready && provision_and_launch_all_workers
        else
            log "Create failed (probably capacity). Will retry next tick."
        fi
        ;;
    CREATING|REPAIRING)
        log "TPU is $state — provisioning in progress. Skipping this tick."
        ;;
    *)
        log "Unknown state '$state'. Skipping."
        ;;
esac
log "--- end tick ---"
