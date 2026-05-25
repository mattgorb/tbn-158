# Running Llama-3

Four sizes × three linear flavors = 12 configs registered in [config_registry.py](torchtitan/models/llama3/config_registry.py). All share the Llama-3 tokenizer (128k vocab) and weight-tied embeddings.

| Size | Architecture                       | `--config` (plain / bitnet158 / tbn158)                                     |
| ---- | ---------------------------------- | --------------------------------------------------------------------------- |
| 200M | dim=768, 16 layers, GQA 12/4       | `llama3_200m`, `llama3_200m_bitnet158`, `llama3_200m_tbn158`                |
| 500M | dim=1024, 28 layers, GQA 16/4      | `llama3_500m`, `llama3_500m_bitnet158`, `llama3_500m_tbn158`                |
| 700M | dim=1536, 24 layers, GQA 24/8      | `llama3_700m`, `llama3_700m_bitnet158`, `llama3_700m_tbn158`                |
| 3B   | dim=3072, 28 layers, GQA 24/8      | `llama3_3b`,   `llama3_3b_bitnet158`,   `llama3_3b_tbn158`                  |

Linear flavors:
- **plain** — standard `nn.Linear` (bf16 baseline)
- **`_bitnet158`** — `BitLinear158`, ternary {-1, 0, +1} weights (arXiv:2402.17764)
- **`_tbn158`** — `TBNBitLinear158`, ternary on tile-averaged weights (tile_size=2)

WandB + checkpointing are on by default.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login           # for the gated Llama-3.2 tokenizer
wandb login                     # paste API key once
export WANDB_PROJECT=tbn158     

python scripts/download_hf_assets.py \
    --repo_id meta-llama/Llama-3.2-3B \
    --local_dir ./assets/hf \
    --assets tokenizer
```

## Train (2 GPUs)

Confirm both GPUs are visible first:

```bash
nvidia-smi -L                                      # expect 2 lines
python -c "import torch; print(torch.cuda.device_count())"   # expect 2
```

If you see more than 2, pin the run to specific devices with `CUDA_VISIBLE_DEVICES=0,1`.

Pick the size you want and run it. Each line is self-contained — change `_tbn158` to `_bitnet158` or drop the suffix for the plain baseline.

```bash
# 200M — fastest iteration, big batch
NGPU=2 MODULE=llama3 CONFIG=llama3_200m_tbn158 ./run_train.sh

# 500M
NGPU=2 MODULE=llama3 CONFIG=llama3_500m_tbn158 ./run_train.sh

# 700M — matches BitNet b1.58 paper size
NGPU=2 MODULE=llama3 CONFIG=llama3_700m_tbn158 ./run_train.sh

# 3B — the big one
NGPU=2 MODULE=llama3 CONFIG=llama3_3b_tbn158 ./run_train.sh
```

Per-size defaults (sized for 80 GB cards):

| Size | local_batch_size | seq_len | lr   | Approx GPU memory |
| ---- | ---------------- | ------- | ---- | ----------------- |
| 200M | 16               | 2048    | 5e-4 | ~12 GB            |
| 500M | 8                | 2048    | 4e-4 | ~20 GB            |
| 700M | 4                | 2048    | 4e-4 | ~22 GB            |
| 3B   | 2                | 4096    | 3e-4 | ~40 GB            |


### If you OOM

Apply in order, each is a smaller hammer than the last. Examples use the 3B config; same flags work for any size:

```bash
# 1. Halve seq_len (biggest single win, halves activation memory)
NGPU=2 MODULE=llama3 CONFIG=llama3_3b_tbn158 ./run_train.sh --training.seq_len=2048

# 2. Drop batch size
NGPU=2 MODULE=llama3 CONFIG=llama3_3b_tbn158 ./run_train.sh --training.local_batch_size=1

# 3. Full activation checkpointing (more recompute, less stored)
NGPU=2 MODULE=llama3 CONFIG=llama3_3b_tbn158 ./run_train.sh --activation_checkpoint.mode=full

# 4. Last resort — FSDP CPU offload (much slower)
NGPU=2 MODULE=llama3 CONFIG=llama3_3b_tbn158 ./run_train.sh --training.enable_cpu_offload
```

Stack flags as needed: `... --training.seq_len=2048 --training.local_batch_size=1 --activation_checkpoint.mode=full`.

## Resume after a crash

Same command. The trainer auto-resumes from the latest `./outputs/checkpoint/step-N/` (interval 500, latest 4 kept). Per-checkpoint size scales with model: ~2 GB at 200M, ~6 GB at 500M, ~9 GB at 700M, ~38 GB at 3B.

To pin a specific step: `--checkpoint.load_step=2500`.

## macOS dry-run

`./run_train.sh`'s shebang points at `/usr/bin/bash` (Linux). On macOS use `bash ./run_train.sh` instead, with `COMM_MODE=fake_backend` to skip NCCL:

```bash
NGPU=2 COMM_MODE=fake_backend MODULE=llama3 CONFIG=llama3_3b_tbn158 bash ./run_train.sh
```
