# tbn-158

Single-GPU pretraining of Llama-3-style models with **BitNet b1.58** or
**TBN-tiled b1.58** linear layers. HuggingFace `Trainer` + streaming C4.
Stable PyTorch, no nightly drama.

## Sizes and variants

Four sizes ([bitnet/model.py](bitnet/model.py)):

| Size  | Architecture                                        |
| ----- | --------------------------------------------------- |
| 200M  | dim=768,  16 layers, GQA 12/4                       |
| 500M  | dim=1024, 28 layers, GQA 16/4                       |
| 700M  | dim=1536, 24 layers, GQA 24/8 *(BitNet paper size)* |
| 3B    | dim=3072, 28 layers, GQA 24/8                       |

Three linear variants ([bitnet/linears.py](bitnet/linears.py)):
- **`plain`** — standard `nn.Linear` (bf16 baseline)
- **`bitnet158`** — ternary {-1, 0, +1} weights, 8-bit activations (arXiv:2402.17764)
- **`tbn158`** — ternary on tile-averaged weights (tile_size=2)

Embeddings and `lm_head` stay full precision in every variant, per the b1.58 paper.

## Setup

```bash
# 1. torch matched to your CUDA (cu124 / cu126 / cu128 / cu130 / cpu)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. everything else
pip install -r requirements.txt

# 3. auth — for the gated Llama-3 tokenizer + WandB
huggingface-cli login
wandb login

# 4. sanity
python -c "from bitnet import build_model; m = build_model('200M', 'tbn158'); print(sum(p.numel() for p in m.parameters()), 'params')"
```

The tokenizer auto-downloads on first run via `AutoTokenizer.from_pretrained` —
no separate download step needed.

## Train

```bash
# smallest, fastest iteration
python pretrain.py --size 200M --variant tbn158

# 700M (BitNet paper size)
python pretrain.py --size 700M --variant bitnet158

# plain baseline for A/B
python pretrain.py --size 200M --variant plain
```

Outputs land in `./outputs/{size}_{variant}/`. WandB runs are named
`{size}_{variant}`; group via env var `WANDB_PROJECT` (default `bitnet158`).

## Per-size defaults (single 48 GB GPU)

| Size  | `--per_device_batch_size` | `--lr`  |
| ----- | ------------------------- | ------- |
| 200M  | 16                        | 5e-4    |
| 500M  | 8                         | 4e-4    |
| 700M  | 4                         | 4e-4    |
| 3B    | 1                         | 3e-4    |

Override anything from the CLI:

```bash
python pretrain.py --size 700M --variant tbn158 \
    --steps 50000 --warmup_steps 1000 \
    --per_device_batch_size 2 --gradient_accumulation_steps 4 \
    --lr 3e-4
```

## Multi-GPU

`Trainer` auto-switches to DDP under `torchrun`. No code changes.

```bash
# all visible GPUs on this node
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) pretrain.py --size 700M --variant tbn158

# specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 pretrain.py --size 700M --variant tbn158
```

`--per_device_batch_size` is **per GPU** — global batch scales with GPU count.

For **3B**, DDP replicates the full model per GPU and will OOM. Run
`accelerate config` once (pick FSDP or DeepSpeed ZeRO-2/3), then:

```bash
accelerate launch pretrain.py --size 3B --variant tbn158
```

## If you OOM

Apply in order:

```bash
# 1. shorter sequences
python pretrain.py ... --seq_len 1024

# 2. smaller batch, recover effective batch with accumulation
python pretrain.py ... --per_device_batch_size 1 --gradient_accumulation_steps 8
```

Gradient checkpointing is on by default. Use `--no_gradient_checkpointing` to
turn it off (faster, more memory).

## Resume after a crash

```bash
# single GPU
python pretrain.py --size 200M --variant tbn158 --resume

# multi-GPU — same launcher as the original run, just add --resume
torchrun --nproc_per_node=4 pretrain.py --size 700M --variant tbn158 --resume
accelerate launch pretrain.py --size 3B --variant tbn158 --resume
```

Picks up from the latest `outputs/{size}_{variant}/checkpoint-N/` and restores
model + optimizer + LR + dataloader state. Default retention: latest 4
checkpoints, saved every 500 steps. Relaunch with the **same world size** you
crashed at — optimizer-state shards (FSDP/ZeRO) are sized to it.

## Notes

- C4 streams from HuggingFace on demand. No upfront download, but needs
  internet during training.
- Stable PyTorch (≥ 2.4) is sufficient. No nightly required.
- Gradient checkpointing on by default; bf16 on by default.
