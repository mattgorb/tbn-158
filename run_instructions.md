# Running Llama-3 (3B)

Three flavors registered in [config_registry.py](torchtitan/models/llama3/config_registry.py):

| `--config`            | Linears                                |
| --------------------- | -------------------------------------- |
| `llama3_3b`           | Standard `nn.Linear`                   |
| `llama3_3b_bitnet158` | `BitLinear158` (ternary {-1, 0, +1})   |
| `llama3_3b_tbn158`    | `TBNBitLinear158`, tile_size=2         |

WandB + checkpointing are on by default.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login           # for the gated Llama-3.2 tokenizer
wandb login                     # paste API key once
export WANDB_PROJECT=tbn158     

python -m torchtitan.tools.download_hf_assets \
    --repo_id meta-llama/Llama-3.2-3B --assets_dir ./assets/hf
```

## Train (2 GPUs)

Confirm both GPUs are visible first:

```bash
nvidia-smi -L                                      # expect 2 lines
python -c "import torch; print(torch.cuda.device_count())"   # expect 2
```

If you see more than 2, pin the run to specific devices with `CUDA_VISIBLE_DEVICES=0,1`.

```bash
NGPU=2 MODULE=llama3 CONFIG=llama3_3b_tbn158 ./run_train.sh
```

Defaults (batch=2, seq=4096, FSDP, selective AC) fit ~35–45 GB/GPU on 80 GB cards. 

If you OOM, apply these in order — each is a smaller hammer than the last:

```bash
# 1. Halve seq_len (biggest single win, halves activation memory)
./run_train.sh ... --training.seq_len=2048

# 2. Drop batch size
./run_train.sh ... --training.local_batch_size=1

# 3. Full activation checkpointing (more recompute, less stored)
./run_train.sh ... --activation_checkpoint.mode=full

# 4. Last resort — FSDP CPU offload (much slower)
./run_train.sh ... --training.enable_cpu_offload
```

Stack them if needed (`--training.seq_len=2048 --training.local_batch_size=1 ...`). Swap `llama3_3b_tbn158` for `llama3_3b_bitnet158` or `llama3_3b` for the other flavors.

## Resume after a crash

Same command. The trainer auto-resumes from the latest `./outputs/checkpoint/step-N/` (interval 500, latest 4 kept, ~38 GB each → ~150 GB total).

To pin a specific step: `--checkpoint.load_step=2500`.

## macOS dry-run

`./run_train.sh`'s shebang points at `/usr/bin/bash` (Linux). On macOS use `bash ./run_train.sh` instead, with `COMM_MODE=fake_backend` to skip NCCL:

```bash
NGPU=2 COMM_MODE=fake_backend MODULE=llama3 CONFIG=llama3_3b_tbn158 bash ./run_train.sh
```
