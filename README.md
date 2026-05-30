# tbn-158

Pretraining of Llama-3-style models with **BitNet b1.58** or **TBN-tiled
b1.58** linear layers. HuggingFace `Trainer` + streaming **FineWeb-Edu**
(train) and **WikiText-103** (eval). Stable PyTorch, no nightly drama.

---

## Setup

Check each host's max CUDA with `nvidia-smi` (top-right cell), then pick the
matching wheel index. Install the torch trio together so versions stay in sync:

- Wheel selector: <https://pytorch.org/get-started/locally/>
- Pinned versions (find `2.6.0`): <https://pytorch.org/get-started/previous-versions/>

```bash
# 1. torch + torchvision + torchaudio, matched trio on your CUDA wheel index
pip install --force-reinstall \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
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

On 2× A100 80 GB:

```bash
bash scripts/train_3b_all.sh
```

This trains all three 3B variants (`plain` → `bitnet158` → `tbn158`)
sequentially via FSDP, ~10B tokens each. Crash-safe: rerun the same command
and each variant resumes from its latest checkpoint. Stdout/stderr is tee'd
to `logs/3B_<variant>.log`.

Or run any variant individually (these are exactly what the sweep script
issues — same flags, same log destination, same `--resume` behavior):

```bash
mkdir -p logs

# plain
accelerate launch --config_file configs/accelerate_fsdp_2gpu.yaml \
    pretrain.py --size 3B --variant plain --resume \
    2>&1 | tee logs/3B_plain.log

# bitnet158
accelerate launch --config_file configs/accelerate_fsdp_2gpu.yaml \
    pretrain.py --size 3B --variant bitnet158 --resume \
    2>&1 | tee logs/3B_bitnet158.log

# tbn158
accelerate launch --config_file configs/accelerate_fsdp_2gpu.yaml \
    pretrain.py --size 3B --variant tbn158 --resume \
    2>&1 | tee logs/3B_tbn158.log
```

**WandB**: open the `bitnet158` project — `train/num_input_tokens_seen`
climbs linearly from 0 to ~10B over each run; `eval/perplexity` is logged
every 1000 steps against the WikiText-103 validation split.

Then evaluate each final checkpoint on the BitNet paper's zero-shot suite
(HellaSwag, WinoGrande, ARC-e/c, PIQA, BoolQ, OBQA, LAMBADA).

3B checkpoints are saved as FSDP sharded state — consolidate first, then eval:

```bash
for V in plain bitnet158 tbn158; do
    bash scripts/consolidate_fsdp.sh outputs/3B_${V}/final
    bash scripts/eval.sh             outputs/3B_${V}/final/consolidated
done
```

### Run on Cloud TPU (v6e-8, single host)

If you have TPU credits (e.g. TRC) and no GPUs handy, the 3B sweep also runs
on a single-host **v6e-8** (8 chips × 32 GB = 256 GB HBM). Wall-clock per
10B-token run: ~22 hours on v6e-8 spot.

**One-time on your laptop** — install gcloud CLI and provision the VM:

```bash
# 1. Install gcloud CLI (macOS):
brew install --cask google-cloud-sdk

# 2. Auth + set project (your project is tbn158, number 759280869534):
gcloud auth login
gcloud config set project tbn158

# 3. Provision a v6e-8 spot VM in us-east1-d:
gcloud compute tpus tpu-vm create matt-tbn158 \
    --zone=us-east1-d \
    --accelerator-type=v6e-8 \
    --version=v2-alpha-tpuv6e \
    --spot

# 4. SSH in:
gcloud compute tpus tpu-vm ssh matt-tbn158 --zone=us-east1-d
```

**On the TPU VM** — clone, install, train:

```bash
git clone https://github.com/<you>/tbn-158.git && cd tbn-158
bash scripts/setup_tpu_vm.sh                  # installs torch_xla + deps
huggingface-cli login
wandb login

bash scripts/train_3b_tpu.sh                  # all 3 variants, ~3 days serial
bash scripts/train_3b_tpu.sh tbn158           # or one variant at a time
```

Then consolidate + eval the same way as the GPU runs.

**For faster runs (v6e-32, 4 hosts, ~6 hours per variant)**: provision with
`--accelerator-type=v6e-32`, and launch with the multi-host fan-out command in
[`scripts/train_3b_tpu.sh`](scripts/train_3b_tpu.sh)'s header (uses
`--worker=all`). v6e-32 is spot-only — if it preempts, the script's `--resume`
flag picks up at the latest checkpoint when you relaunch.

**Spot preemption survival via GCS** (strongly recommended for any TPU run > a
few hours): mount a GCS bucket on the VM and write checkpoints there instead
of local disk. When the VM is preempted, just spin up a new one and re-mount —
`--resume` picks up at the latest checkpoint from the bucket.

```bash
# 1. ONE-TIME (from laptop) — create a bucket in or near your TPU region:
gcloud storage buckets create gs://matt-tbn158-ckpts --location=us-east1
# (or --location=europe-west4 if your TPU is in europe-west4-a)

# 2. ON EVERY VM (gcsfuse is auto-installed by setup_tpu_vm.sh) —
#    just set GCS_BUCKET when launching:
GCS_BUCKET=matt-tbn158-ckpts bash scripts/train_3b_tpu.sh

# 3. AFTER PREEMPTION — spin up a new VM, run setup, then:
GCS_BUCKET=matt-tbn158-ckpts bash scripts/train_3b_tpu.sh
# The script remounts the bucket and resume picks up where you left off.
```

If you omit `GCS_BUCKET`, checkpoints go to local disk and **die with the VM
on preemption**. Don't omit it for long runs.

### Paper-scale run (4× A100 80 GB, 100B tokens, tbn158 only)

If you have 4 GPUs and want to match the BitNet paper's training-token budget:

```bash
accelerate launch --config_file configs/accelerate_fsdp_4gpu.yaml \
    pretrain.py --size 3B --variant tbn158 \
    --config configs/3B_tbn158_4gpu_100b.yaml --resume
```

100K steps × 1M tokens/step ≈ **105B tokens**. Expect multi-week wall-clock.

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

## Eval

**During training** — held-out WikiText-103 perplexity logged to WandB every
`eval_steps` (default 1000) over `eval_samples` sequences (default 512). Logged
keys: `eval/loss`, `eval/perplexity`, `train/num_input_tokens_seen`. Set
`eval_samples: 0` to disable.

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

## Notes

- FineWeb-Edu streams from HuggingFace on demand. No upfront download, but needs
  internet during training.
- Stable PyTorch (≥ 2.6) is sufficient. No nightly required.
- Gradient checkpointing on by default; bf16 on by default.
