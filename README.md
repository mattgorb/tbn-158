# tbn-158

Pretraining of Llama-3-style models with **BitNet b1.58** or **TBN-tiled
b1.58** linear layers. HuggingFace `Trainer` + streaming **SlimPajama**.
Stable PyTorch, no nightly drama.

---

## Smoke test first (30-60 seconds)

Before any long run, verify end-to-end training works on the 200M model:

```bash
bash scripts/smoke_test.sh           # tbn158 by default
bash scripts/smoke_test.sh plain     # or another variant
```

Runs 1 optimizer step + 1 full eval round, saves a checkpoint, and confirms
WandB logging works. Outputs go to `outputs/smoke_test/` (won't pollute real runs).

---

## 🚀 Run the 3B sweep (the main thing)

After [Setup](#setup), on 2× A100 80 GB:

```bash
bash scripts/train_3b_all.sh
```

This trains all three 3B variants (`plain` → `bitnet158` → `tbn158`)
sequentially via FSDP, ~10B tokens each. Crash-safe: rerun the same command
and each variant resumes from its latest checkpoint. Stdout/stderr is tee'd
to `logs/3B_<variant>.log`.

**WandB**: open the `bitnet158` project — `train/num_input_tokens_seen`
climbs linearly from 0 to ~10B over each run; `eval/perplexity` is logged
every 1000 steps against the SlimPajama validation split.

Then evaluate each final checkpoint on the BitNet paper's zero-shot suite
(HellaSwag, WinoGrande, ARC-e/c, PIQA, BoolQ, OBQA, LAMBADA):

```bash
bash scripts/eval.sh outputs/3B_plain/final
bash scripts/eval.sh outputs/3B_bitnet158/final
bash scripts/eval.sh outputs/3B_tbn158/final
```

---

## Sizes and variants

Three sizes ([bitnet/model.py](bitnet/model.py)):

| Size  | Architecture                                        |
| ----- | --------------------------------------------------- |
| 200M  | dim=768,  16 layers, GQA 12/4                       |
| 700M  | dim=1536, 24 layers, GQA 24/8 *(BitNet paper size)* |
| 3B    | dim=3072, 28 layers, GQA 24/8                       |

Three linear variants ([bitnet/linears.py](bitnet/linears.py)):
- **`plain`** — standard `nn.Linear` (bf16 baseline)
- **`bitnet158`** — ternary {-1, 0, +1} weights, 8-bit activations (arXiv:2402.17764)
- **`tbn158`** — ternary on tile-averaged weights (tile_size=2)

Embeddings and `lm_head` stay full precision in every variant, per the b1.58 paper.

## Setup

Check each host's max CUDA with `nvidia-smi` (top-right cell), then pick the
matching wheel index. Install the torch trio together so versions stay in sync:

- Wheel selector: <https://pytorch.org/get-started/locally/>
- Pinned versions (find `2.5.1`): <https://pytorch.org/get-started/previous-versions/>

```bash
# 1. torch + torchvision + torchaudio, matched trio on your CUDA wheel index
pip install --force-reinstall \
    "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cu124

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

## Train — recommended runs

Configs in [`configs/`](configs/) are tuned to a target token budget per size.
Pick a `<variant>` ∈ `{plain, bitnet158, tbn158}` and run the matching launcher.
Outputs land in `./outputs/{size}_{variant}/`.

```bash
# 200M — 1× 48 GB GPU, ~2.1B tokens (8000 steps × 262K tokens/step)
python pretrain.py --size 200M --variant <variant>

# 700M — 2× A100 80 GB (DDP), ~10B tokens (38000 steps)
torchrun --nproc_per_node=2 pretrain.py --size 700M --variant <variant>

# 3B — see `scripts/train_3b_all.sh` (FSDP via accelerate, all 3 variants)
```

Token budget summary (global batch is 128 sequences × 2048 tokens = 262K tokens/step everywhere):

| Size  | Hardware            | Launcher              | Steps  | Tokens |
| ----- | ------------------- | --------------------- | ------ | ------ |
| 200M  | 1× 48 GB            | `python`              | 8,000  | ~2.1 B |
| 700M  | 2× A100 80 GB       | `torchrun --nproc 2`  | 38,000 | ~10 B  |
| 3B    | 2× A100 80 GB FSDP  | `scripts/train_3b_all.sh` | 38,000 | ~10 B  |

WandB runs are named `{size}_{variant}`; group via `WANDB_PROJECT` (default
`bitnet158`).

## Eval

**During training** — held-out SlimPajama perplexity logged to WandB every
`eval_steps` (default 1000) over `eval_samples` sequences (default 512, ≈ 1M
tokens). Logged keys: `eval/loss`, `eval/perplexity`,
`train/num_input_tokens_seen`. Set `eval_samples: 0` to disable.

**Post-hoc zero-shot** — [`scripts/eval.sh`](scripts/eval.sh) runs the BitNet
paper's benchmark suite (HellaSwag, WinoGrande, ARC-e/c, PIQA, BoolQ, OBQA,
LAMBADA) via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
bash scripts/eval.sh outputs/700M_tbn158/final
```

Results land in `<ckpt>/eval_results/`.

## Configs

Per-(size, variant) hyperparams live in [`configs/`](configs/) as YAML — e.g.
[`configs/700M_tbn158.yaml`](configs/700M_tbn158.yaml). `pretrain.py`
auto-loads `configs/{size}_{variant}.yaml`; override with `--config path.yaml`.

Precedence: **CLI flag > YAML > built-in fallback**. Edit the YAML for
persistent changes; pass flags for one-off overrides:

```bash
python pretrain.py --size 700M --variant tbn158 \
    --per_device_batch_size 1 --gradient_accumulation_steps 32
```

`--per_device_batch_size` is **per GPU** under `torchrun` / `accelerate`, so
global batch = `per_device_batch_size × num_gpus × gradient_accumulation_steps`.

## If you OOM

Edit `configs/{size}_{variant}.yaml` (persistent) or pass flags (one-off).
Apply in order:

```yaml
# 1. smaller batch, recover effective batch with accumulation
per_device_batch_size: 1
gradient_accumulation_steps: 8

# 2. shorter sequences
seq_len: 1024
```

Gradient checkpointing is on by default. Set `no_gradient_checkpointing: true`
to turn it off (faster, more memory).

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

- SlimPajama streams from HuggingFace on demand. No upfront download, but needs
  internet during training.
- Stable PyTorch (≥ 2.4) is sufficient. No nightly required.
- Gradient checkpointing on by default; bf16 on by default.
