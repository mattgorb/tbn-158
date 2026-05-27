# tbn-158 — BitNet b1.58 pretraining

Single-GPU pretraining of Llama-3-style models with BitNet b1.58 / TBN-tiled
b1.58 linear layers. Built on HuggingFace `transformers` + `Trainer`.

## Layout
- [`bitnet/linears.py`](bitnet/linears.py) — `BitLinear158`, `TBNBitLinear158`
  (subclasses of `nn.Linear`, STE in forward, full-precision master weights).
- [`bitnet/model.py`](bitnet/model.py) — Llama-3 configs (200M / 700M / 3B;
  500M arch defined but no shipped config) and the linear-swap helper.
- [`pretrain.py`](pretrain.py) — single entrypoint; HF `Trainer` streaming
  FineWeb-Edu (train) + WikiText-103 (validation PPL).
- [`configs/`](configs/) — per-(size, variant) YAML training hyperparams; auto-
  loaded by `pretrain.py` as `configs/{size}_{variant}.yaml`. CLI flags override.
- [`scripts/train_3b_all.sh`](scripts/train_3b_all.sh) — sequential 3B sweep
  (plain → bitnet158 → tbn158) on 2× A100 80 GB via FSDP.
- [`scripts/eval.sh`](scripts/eval.sh) — post-hoc lm-evaluation-harness on a
  saved checkpoint.
- [`requirements.txt`](requirements.txt) — stable torch + HF stack; no nightly required.

## Conventions
- Embeddings and `lm_head` stay full precision in all variants (b1.58 paper rule).
- Default to bf16 + gradient checkpointing; override via `pretrain.py` flags.
- Don't reintroduce torchtitan deps. If a feature needs distributed training
  beyond `accelerate`'s built-in support, justify it.
- Per the b1.58 paper, `gamma = mean(|W|)` is computed over the full weight
  tensor. Don't change this without good reason — and if you ever add TP,
  remember to all-reduce `gamma` across the TP group.

## Don't
- Don't add nightly torch as a requirement.
- Don't add new `nn.Linear` swaps without matching the STE pattern in the
  existing classes (forward sees `w + (w_q - w).detach()`).
