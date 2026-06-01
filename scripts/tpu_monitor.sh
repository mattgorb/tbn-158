#!/usr/bin/env bash
# Cron-friendly TPU watchdog. Runs periodically (e.g. every 15 min), ensures
# the TPU exists and training is alive. If the TPU is preempted, recreates it
# (trying multiple zones in order), runs setup on every worker, and launches
# training in detached tmux on every worker (resumes from GCS via --resume).
#
# Multi-zone aware: ZONES is a comma-separated list. The script searches every
# zone for the TPU, and on (re)create tries them in order until one accepts.

set -uo pipefail

# ============ CONFIG ============
TPU_NAME="${TPU_NAME:-matt-tbn158-2}"
# Comma-separated zones, tried in priority order. The script searches all of
# them every tick to find where the TPU actually lives.
ZONES="${ZONES:-us-east1-d,europe-west4-a}"
ACCEL_TYPE="${ACCEL_TYPE:-v6e-8}"
VERSION="${VERSION:-v2-alpha-tpuv6e}"
NETWORK="${NETWORK:-tbn158-net}"
REPO_URL="${REPO_URL:?REPO_URL env var required, e.g. https://<pat>@github.com/<you>/tbn-158.git}"
GCS_BUCKET="${GCS_BUCKET:-matt-tbn158-ckpts}"
VARIANT="${VARIANT:-tbn158}"
USE_SPOT="${USE_SPOT:-true}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_tpu_v6e_8.yaml}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
CONFIG_FILE="${CONFIG_FILE:-}"

HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required}"
WANDB_TOKEN="${WANDB_TOKEN:?WANDB_TOKEN env var required}"

LOG_FILE="${LOG_FILE:-/tmp/tpu_monitor.log}"

IFS=',' read -ra ZONE_LIST <<< "$ZONES"

# ============ SSH KEY (cron-safe) ============
mkdir -p "$HOME/.ssh"
if [ ! -f "$HOME/.ssh/google_compute_engine" ]; then
    ssh-keygen -t rsa -f "$HOME/.ssh/google_compute_engine" -N "" -q
fi
if ! ssh-keygen -y -P "" -f "$HOME/.ssh/google_compute_engine" >/dev/null 2>&1; then
    echo "ERROR: ~/.ssh/google_compute_engine has a passphrase. Delete and retry:" >&2
    echo "  rm -f ~/.ssh/google_compute_engine{,.pub} && ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -N '' -q" >&2
    exit 1
fi

# ============ HELPERS ============
# Log to stderr (so functions can `echo` return values to stdout for capture).
log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG_FILE" >&2; }

# Search every configured zone for the TPU. Echoes "STATE ZONE" if found, or
# "NOT_FOUND ''" if absent everywhere.
find_tpu() {
    for z in "${ZONE_LIST[@]}"; do
        local s
        s=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$z" \
            --format='value(state)' 2>/dev/null)
        if [ -n "$s" ]; then
            echo "$s $z"
            return
        fi
    done
    echo "NOT_FOUND "
}

training_is_running() {
    local zone="$1"
    local count
    log "Checking if training process exists on worker 0..."
    # 30s timeout — quick check. If SSH key propagation is slow, fall through
    # to the relaunch path (which is a safe no-op if training was actually up).
    count=$(timeout 30 gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$zone" --worker=0 \
        --command='pgrep -fc pretrain.py || echo 0' 2>/dev/null | tr -d '[:space:]')
    log "training_is_running probe returned: '${count:-<empty>}'"
    [ "${count:-0}" -gt 0 ]
}

# Try every configured zone in order. Echoes the successful zone, returns 0.
# If all zones fail, returns 1 and echoes empty.
create_tpu_in_any_zone() {
    local spot_flag=""
    [ "$USE_SPOT" = "true" ] && spot_flag="--spot"
    for z in "${ZONE_LIST[@]}"; do
        log "Attempting create in zone $z..."
        if gcloud compute tpus tpu-vm create "$TPU_NAME" --zone="$z" \
            --accelerator-type="$ACCEL_TYPE" --version="$VERSION" \
            --network="$NETWORK" $spot_flag 2>&1 | tee -a "$LOG_FILE" >&2; then
            log "Successfully created $TPU_NAME in $z"
            echo "$z"
            return 0
        fi
        log "Create failed in $z, trying next zone..."
    done
    log "All zones exhausted. Will retry next tick."
    return 1
}

wait_for_ready() {
    local zone="$1"
    log "Waiting for TPU to become READY in $zone (polling every 15s, max 15 min)..."
    for i in $(seq 1 60); do
        local s
        s=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$zone" \
            --format='value(state)' 2>/dev/null || echo "")
        if [ "$s" = "READY" ]; then
            log "TPU is READY in $zone (after $((i * 15))s)."
            return 0
        fi
        log "  poll #$i: state=$s, sleeping 15s..."
        sleep 15
    done
    log "TPU did not become READY within 15 min. Aborting this cycle."
    return 1
}

provision_and_launch_all_workers() {
    local zone="$1"
    log "Provisioning + launching across all workers of $TPU_NAME (zone=$zone)..."
    # 20 min cap. Echo markers below let you see which phase is running.
    timeout 1200 gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$zone" --worker=all \
        --command="
            set -e
            echo '[provision] === START ==='
            if [ ! -d \$HOME/tbn-158 ]; then
                echo '[provision] cloning repo...'
                git clone $REPO_URL \$HOME/tbn-158
                cd \$HOME/tbn-158
                echo '[provision] running setup_tpu_vm.sh (pip installs, ~5-10 min)...'
                bash scripts/setup_tpu_vm.sh
                echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> \$HOME/.bashrc
                echo '[provision] installing tmux...'
                sudo apt install -y tmux
            else
                echo '[provision] repo already cloned, skipping setup'
            fi
            export PATH=\"\$HOME/.local/bin:\$PATH\"
            echo '[provision] hf + wandb auth...'
            hf auth login --token $HF_TOKEN 2>/dev/null || huggingface-cli login --token $HF_TOKEN 2>/dev/null || true
            wandb login $WANDB_TOKEN 2>/dev/null || true
            echo '[provision] git pull latest...'
            cd \$HOME/tbn-158 && git pull
            tmux kill-session -t train 2>/dev/null || true
            echo '[provision] launching tmux session train...'
            tmux new -d -s train \"ACCEL=$ACCEL_CONFIG GCS_BUCKET=$GCS_BUCKET RUN_SUFFIX=$RUN_SUFFIX CONFIG_FILE=$CONFIG_FILE bash scripts/train_3b_tpu.sh $VARIANT 2>&1 | tee /tmp/train.log\"
            echo '[provision] === DONE ==='
        " 2>&1 | tee -a "$LOG_FILE"
}

# ============ MAIN ============
log "--- tick ($TPU_NAME zones=$ZONES) ---"
read -r state found_zone < <(find_tpu)
log "TPU state: $state${found_zone:+ (zone=$found_zone)}"

case "$state" in
    READY)
        if training_is_running "$found_zone"; then
            log "Training is running on worker 0 in $found_zone. Nothing to do."
        else
            log "TPU up in $found_zone but no training process. Relaunching..."
            provision_and_launch_all_workers "$found_zone"
        fi
        ;;
    PREEMPTED|STOPPED|TERMINATED)
        log "TPU is $state in $found_zone. Deleting + recreating across zones..."
        gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$found_zone" --quiet 2>/dev/null || true
        if new_zone=$(create_tpu_in_any_zone); then
            wait_for_ready "$new_zone" && provision_and_launch_all_workers "$new_zone"
        fi
        ;;
    NOT_FOUND)
        log "TPU not found in any zone. Creating..."
        if new_zone=$(create_tpu_in_any_zone); then
            wait_for_ready "$new_zone" && provision_and_launch_all_workers "$new_zone"
        fi
        ;;
    CREATING|REPAIRING)
        log "TPU is $state in $found_zone — provisioning in progress. Skipping this tick."
        ;;
    *)
        log "Unknown state '$state' in $found_zone. Skipping."
        ;;
esac
log "--- end tick ---"
