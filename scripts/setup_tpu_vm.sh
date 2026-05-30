#!/usr/bin/env bash
# Runs ON the TPU VM (after `gcloud compute tpus tpu-vm ssh ...`).
# Installs torch_xla + repo deps, clones the repo, and prints next-step auth.
#
# Usage on the TPU VM:
#   curl -sSL https://raw.githubusercontent.com/<your>/<repo>/main/scripts/setup_tpu_vm.sh | bash
# OR after cloning manually:
#   bash scripts/setup_tpu_vm.sh

set -euo pipefail

echo "==> Installing PyTorch + torch_xla (TPU build) ..."
pip install --upgrade pip
pip install "torch~=2.6.0" "torch_xla[tpu]~=2.6.0" \
    -f https://storage.googleapis.com/libtpu-releases/index.html

echo "==> Installing project dependencies ..."
pip install "transformers>=4.50" "accelerate>=1.0" "datasets>=3.0" \
    "tokenizers>=0.15" "huggingface_hub>=0.25" \
    wandb tensorboard einops safetensors pyyaml "lm-eval>=0.4"

echo "==> Sanity-checking torch_xla can see the TPUs ..."
python -c "import torch_xla.core.xla_model as xm; print('xla_device:', xm.xla_device()); print('world_size:', xm.xla_real_devices())"

echo "==> Installing gcsfuse (for GCS-backed checkpoint persistence) ..."
if ! command -v gcsfuse >/dev/null 2>&1; then
    export GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)"
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" \
        | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    sudo apt-get update -y
    sudo apt-get install -y gcsfuse
fi
echo "    gcsfuse: $(gcsfuse --version | head -1)"

cat <<'EOF'

==> Next steps:
  huggingface-cli login    # Llama-3 tokenizer is gated
  wandb login              # for run logging

  cd ~/tbn-158 || git clone https://github.com/<you>/tbn-158.git ~/tbn-158 && cd ~/tbn-158

  bash scripts/train_3b_tpu.sh           # all 3 variants sequentially
  bash scripts/train_3b_tpu.sh tbn158    # one variant
EOF
