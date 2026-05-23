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
export WANDB_PROJECT=tbn158     # optional; group runs in the UI

python -m torchtitan.tools.download_hf_assets \
    --repo_id meta-llama/Llama-3.2-3B --assets_dir ./assets/hf
```

## Train (2 GPUs)

```bash
NGPU=2 MODULE=llama3 CONFIG=llama3_3b_tbn158 ./run_train.sh
```

Swap `llama3_3b_tbn158` for `llama3_3b_bitnet158` or `llama3_3b` for the other flavors. Override anything from the CLI: `--training.steps=20000`, `--optimizer.lr=2e-4`, `--training.local_batch_size=1`, etc.

## Resume after a crash

Same command. The trainer auto-resumes from the latest `./outputs/checkpoint/step-N/` (interval 500, latest 10 kept).

To pin a specific step: `--checkpoint.load_step=2500`.

## macOS dry-run

`./run_train.sh`'s shebang points at `/usr/bin/bash` (Linux). On macOS use `bash ./run_train.sh` instead, with `COMM_MODE=fake_backend` to skip NCCL:

```bash
NGPU=2 COMM_MODE=fake_backend MODULE=llama3 CONFIG=llama3_3b_tbn158 bash ./run_train.sh
```
