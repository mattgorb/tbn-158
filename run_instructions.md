# BitNet / TBN Pretraining

Pretrain a Llama-3-style model with optional BitNet b1.58 or TBN-tiled b1.58
linears, on a single GPU, using HuggingFace `Trainer`. Streaming C4, automatic
checkpointing, WandB + TensorBoard.

The torchtitan code in this repo is no longer the recommended entry point;
[`pretrain.py`](pretrain.py) + [`bitnet/`](bitnet/) is. Torchtitan stays in the
tree for reference but is not used by this workflow.

## Sizes and variants

Four sizes (defined in [bitnet/model.py](bitnet/model.py)):

| Size  | Architecture                                        |
| ----- | --------------------------------------------------- |
| 200M  | dim=768,  16 layers, GQA 12/4                       |
| 500M  | dim=1024, 28 layers, GQA 16/4                       |
| 700M  | dim=1536, 24 layers, GQA 24/8 *(BitNet paper size)* |
| 3B    | dim=3072, 28 layers, GQA 24/8                       |

Three linear variants (from [bitnet/linears.py](bitnet/linears.py)):
- **`plain`** — standard `nn.Linear` (bf16 baseline)
- **`bitnet158`** — ternary {-1, 0, +1} weights, 8-bit activations (arXiv:2402.17764)
- **`tbn158`** — ternary on tile-averaged weights (tile_size=2)

## Setup

```bash
# 1. Install torch matched to your CUDA (cu124 / cu126 / cu128 / cu130 / cpu)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. Everything else
pip install -r requirements.txt

# 3. Auth — for the gated Llama-3 tokenizer + WandB
huggingface-cli login
wandb login

# 4. Sanity check
python -c "from bitnet import build_model; m = build_model('200M', 'tbn158'); print(sum(p.numel() for p in m.parameters()), 'params')"
```

## Train

```bash
# Smallest, fastest iteration
python pretrain.py --size 200M --variant tbn158

# 700M (BitNet paper size)
python pretrain.py --size 700M --variant bitnet158

# Plain baseline at any size for A/B comparison
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

## If you OOM

Apply in order, each is a smaller hammer than the last:

```bash
# 1. Shorter sequences (halves activation memory)
python pretrain.py ... --seq_len 1024

# 2. Drop batch size, recover with grad accumulation if you want the same effective batch
python pretrain.py ... --per_device_batch_size 1 --gradient_accumulation_steps 8

# 3. Gradient checkpointing is on by default; disable only if you have memory headroom and want speed
python pretrain.py ... --no_gradient_checkpointing   # opposite direction (uses MORE memory, faster)
```

## Resume after a crash

```bash
python pretrain.py --size 200M --variant tbn158 --resume
```

Picks up from the latest `outputs/{size}_{variant}/checkpoint-N/` and restores
model + optimizer + LR + dataloader state. To pin a specific step, point
`--resume_from_checkpoint` at the directory manually (set via the underlying
Trainer API in [pretrain.py](pretrain.py)).

Default retention: keep latest 4 checkpoints (`--save_total_limit 4`). Save
every 500 steps.

## Notes

- C4 streams from HuggingFace on demand — no upfront dataset download, but
  needs internet during training.
- Gradient checkpointing is on by default. Disable with
  `--no_gradient_checkpointing` if you have memory to spare and want speed.
- All embeddings and the final `lm_head` stay full precision regardless of
  variant, per the b1.58 paper.
- Stable PyTorch (≥ 2.4) is sufficient. No torch nightly required.
