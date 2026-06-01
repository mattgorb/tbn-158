"""Pretrain a Llama-3-style model with optional BitNet b1.58 / TBN b1.58
linears, on streaming FineWeb-Edu. Single-GPU friendly; uses HuggingFace Trainer.

Usage
-----
    # Smallest model, TBN-tiled b1.58 — uses configs/200M_tbn158.yaml:
    python pretrain.py --size 200M --variant tbn158

    # Override any config value from the CLI (precedence: CLI > YAML > built-in):
    python pretrain.py --size 700M --variant tbn158 --per_device_batch_size 1

    # Point at an explicit config file:
    python pretrain.py --size 200M --variant tbn158 --config configs/my_run.yaml

    # Resume from the latest checkpoint in --output_dir:
    python pretrain.py --size 200M --variant tbn158 --resume

All four sizes (200M / 500M / 700M / 3B) and three variants (plain /
bitnet158 / tbn158) are supported. Per-(size, variant) defaults live in
configs/{size}_{variant}.yaml — edit there for persistent OOM tweaks.
"""

import argparse
import math
import os
from itertools import chain
from pathlib import Path

# Pin tokenizer parallelism before any worker is forked. Avoids segfaults from
# HF Rust tokenizer's threadpool fighting with DataLoader's multiprocessing.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# On TPU, enable XLA SPMD BEFORE any device init — required by HF Trainer's
# xla_fsdp_v2 path. Must run before importing transformers (which can
# implicitly touch the XLA runtime via accelerate).
if os.environ.get("PJRT_DEVICE") == "TPU":
    try:
        import torch_xla.runtime as xr

        xr.use_spmd()
    except ImportError:
        pass

import yaml
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from bitnet import LINEAR_FACTORIES, SIZE_HPARAMS, build_model


# Final fallback if no YAML is found and no CLI flag is given.
BUILTIN_DEFAULTS = {
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "seq_len": 2048,
    "steps": 10000,
    "warmup_steps": 500,
    "lr": 3e-4,
    "weight_decay": 0.1,
    "save_steps": 500,
    "save_total_limit": 4,
    "logging_steps": 10,
    "no_gradient_checkpointing": False,
    # C4 validation perplexity. Set eval_samples: 0 to disable.
    "eval_steps": 1000,
    "eval_samples": 512,
    # bf16 via torch.cuda.amp. Set to false on TPU/XLA — accelerate handles
    # mixed precision via `mixed_precision: bf16` in the accelerate config.
    "bf16": True,
}

# Fields that may appear in YAML configs and as CLI overrides.
OVERRIDABLE = set(BUILTIN_DEFAULTS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--size", choices=list(SIZE_HPARAMS), required=True)
    p.add_argument("--variant", choices=list(LINEAR_FACTORIES), required=True)
    p.add_argument(
        "--config",
        default=None,
        help="YAML config path. Default: configs/{size}_{variant}.yaml if it exists.",
    )
    p.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.2-3B",
        help="HF repo ID for the tokenizer (gated; needs huggingface-cli login).",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Default: ./outputs/{size}_{variant}/",
    )
    p.add_argument("--wandb_project", default="bitnet158")
    p.add_argument("--run_name", default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from the latest checkpoint in --output_dir.",
    )

    # Overridable fields — use SUPPRESS so we can tell whether the user
    # passed them explicitly. Anything omitted from CLI falls back to the
    # YAML config, then to BUILTIN_DEFAULTS.
    p.add_argument("--seq_len", type=int, default=argparse.SUPPRESS)
    p.add_argument("--per_device_batch_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--gradient_accumulation_steps", type=int, default=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, default=argparse.SUPPRESS)
    p.add_argument("--warmup_steps", type=int, default=argparse.SUPPRESS)
    p.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    p.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS)
    p.add_argument("--save_steps", type=int, default=argparse.SUPPRESS)
    p.add_argument("--save_total_limit", type=int, default=argparse.SUPPRESS)
    p.add_argument("--logging_steps", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_steps", type=int, default=argparse.SUPPRESS)
    p.add_argument(
        "--eval_samples",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of C4-validation sequences for held-out PPL. 0 disables eval.",
    )
    p.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable gradient checkpointing (faster, more memory).",
    )
    p.add_argument(
        "--bf16",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=argparse.SUPPRESS,
        help="bf16 via CUDA AMP. Set to false on TPU/XLA (accelerate handles it).",
    )
    return p.parse_args()


def load_config(path: str | None, size: str, variant: str) -> dict:
    if path is None:
        default_path = Path(f"configs/{size}_{variant}.yaml")
        if not default_path.exists():
            return {}
        path = str(default_path)
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    unknown = set(loaded) - OVERRIDABLE
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return loaded


def resolve_settings(args: argparse.Namespace) -> dict:
    """Merge precedence: CLI explicit > YAML config > BUILTIN_DEFAULTS."""
    config = load_config(args.config, args.size, args.variant)
    cli_explicit = {k: v for k, v in vars(args).items() if k in OVERRIDABLE}
    return {**BUILTIN_DEFAULTS, **config, **cli_explicit}


def build_dataset(tokenizer, seq_len: int, split: str = "train", take: int | None = None):
    """Stream training/validation data, tokenize, and pack into seq_len blocks.

    Train: HuggingFaceFW/fineweb-edu (sample-350BT subset — covers all budgets).
    Val:   wikitext-103-raw-v1 validation — classic PPL benchmark.
    """
    if split == "train":
        raw = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-350BT",
            split="train",
            streaming=True,
        )
    else:
        raw = load_dataset(
            "Salesforce/wikitext",
            name="wikitext-103-raw-v1",
            split=split,
            streaming=True,
        )

    original_cols = list(raw.features.keys())

    def tokenize(batch):
        return tokenizer(batch["text"])

    tokenized = raw.map(tokenize, batched=True, remove_columns=original_cols)

    def group_into_blocks(batch):
        concatenated = {k: list(chain(*batch[k])) for k in batch}
        total = (len(concatenated["input_ids"]) // seq_len) * seq_len
        result = {
            k: [v[i : i + seq_len] for i in range(0, total, seq_len)]
            for k, v in concatenated.items()
        }
        result["labels"] = [list(ids) for ids in result["input_ids"]]
        return result

    packed = tokenized.map(group_into_blocks, batched=True)
    return packed.take(take) if take else packed


def manual_load_checkpoint_weights(model, ckpt_dir: str) -> int:
    """Load checkpoint weights into the model BEFORE FSDP wraps it.

    Reason: HF Trainer's `_load_from_checkpoint` deadlocks on XLA FSDPv2 SPMD
    because the post-wrap model-load requires an all-ranks broadcast collective
    that only rank 0 actually enters. Pre-loading on the CPU-side model
    (which all 8 XLA workers do) sidesteps the issue — FSDP then shards the
    already-loaded weights normally on first forward.

    Returns the global_step from the checkpoint, or 0 if loading was skipped.
    """
    import json

    # Find the model weights file:
    model_file = None
    for candidate in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(ckpt_dir, candidate)
        if os.path.exists(p):
            model_file = p
            break
    if not model_file:
        print(f"  no model file found in {ckpt_dir}, starting fresh")
        return 0

    print(f"  loading weights from {model_file}...")
    if model_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(model_file)
    else:
        import torch
        state_dict = torch.load(model_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} keys missing from checkpoint, e.g. {missing[:3]}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    del state_dict

    # Read global_step:
    state_file = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            ts = json.load(f)
        gs = ts.get("global_step", 0)
        print(f"  resumed at global_step={gs}")
        return gs
    return 0


class TrainerWithPerplexity(Trainer):
    """Trainer that also logs `eval_perplexity = exp(eval_loss)`.

    Also patches a bug in HF Trainer's FSDP+XLA path where `_prepare_for_training`
    calls `create_scheduler` without first calling `create_optimizer`, leaving
    `self.optimizer = None` and crashing the scheduler on `param_groups`. We
    defensively create the optimizer here if it doesn't exist yet.

    And: overrides `_load_from_checkpoint` to no-op the model load, because
    we pre-load weights in main() before FSDP wraps the model (HF's load path
    deadlocks under XLA FSDPv2 SPMD).
    """

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # Model weights are already loaded by manual_load_checkpoint_weights()
        # in main(). HF Trainer's load would deadlock under XLA FSDPv2 — skip it.
        # Trainer state (global_step, epoch) is still restored from
        # trainer_state.json by the rest of HF Trainer's resume flow.
        return

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if optimizer is None and self.optimizer is None:
            self.create_optimizer()
        return super().create_scheduler(num_training_steps, optimizer or self.optimizer)

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        if "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"])
            metrics["eval_perplexity"] = ppl
            self.log({"eval_perplexity": ppl})
        return metrics


def find_latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(ckpts[-1]) if ckpts else None


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)

    output_dir = args.output_dir or f"./outputs/{args.size}_{args.variant}"
    run_name = args.run_name or f"{args.size}_{args.variant}"

    print(f"Building model: {args.size} / {args.variant}")
    model = build_model(args.size, args.variant)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {num_params:,}")
    print("Settings:")
    for k in sorted(settings):
        print(f"  {k}: {settings[k]}")

    # Manual checkpoint pre-load (XLA FSDPv2 workaround — see docstring on
    # manual_load_checkpoint_weights). Must happen BEFORE Trainer is created
    # so the load goes into the CPU model, before FSDP wraps it.
    resume_from = find_latest_checkpoint(output_dir) if args.resume else None
    if resume_from:
        print(f"Pre-loading checkpoint weights from {resume_from}")
        manual_load_checkpoint_weights(model, resume_from)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Streaming FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-350BT)")
    train_ds = build_dataset(tokenizer, settings["seq_len"], split="train")

    eval_ds = None
    if settings["eval_samples"] > 0:
        print(f"Streaming WikiText-103 validation for held-out PPL ({settings['eval_samples']} seqs)")
        eval_ds = build_dataset(
            tokenizer, settings["seq_len"], split="validation", take=settings["eval_samples"],
        )

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # On TPU/XLA, configure FSDP via TrainingArguments (accelerate's XLA cluster
    # config doesn't accept fsdp keys). FSDPv2 uses XLA SPMD under the hood.
    #
    # NOTE: with FSDP full_shard, Trainer's `gradient_checkpointing=True` is
    # incompatible (HF issue #30404) — it leaves the optimizer uninitialized and
    # the LR scheduler crashes on `optimizer.param_groups`. Put activation
    # checkpointing inside `fsdp_config` instead, and force the Trainer flag off.
    fsdp_kwargs = {}
    on_tpu = os.environ.get("PJRT_DEVICE") == "TPU"
    if on_tpu:
        fsdp_kwargs = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                "xla": True,
                "xla_fsdp_v2": True,
                # XLA-specific key — `activation_checkpointing` is the non-XLA
                # name and silently no-ops here, causing HBM OOM.
                "xla_fsdp_grad_ckpt": not settings["no_gradient_checkpointing"],
            },
        }
        settings = {**settings, "no_gradient_checkpointing": True}
        print("Detected TPU; enabling XLA FSDPv2 sharding via TrainingArguments")

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=settings["per_device_batch_size"],
        per_device_eval_batch_size=settings["per_device_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        max_steps=settings["steps"],
        warmup_steps=settings["warmup_steps"],
        learning_rate=settings["lr"],
        weight_decay=settings["weight_decay"],
        lr_scheduler_type="cosine",
        logging_steps=settings["logging_steps"],
        save_steps=settings["save_steps"],
        save_total_limit=settings["save_total_limit"],
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=settings["eval_steps"],
        # HF Trainer's `_save_optimizer_and_scheduler` deadlocks under
        # xla_fsdp_v2: it gates the save on `if should_save` (rank-0 only),
        # but `xm.save` inside that branch needs a cross-rank gather of the
        # sharded optimizer tensors. Ranks 1-N skip the branch, never join the
        # gather, rank 0 hangs forever. Workaround: skip optimizer-state save
        # entirely on TPU. Model weights + global_step still persist; LR
        # scheduler reconstructs to current step on resume; only Adam moments
        # reset (couple hundred steps of suboptimality after each resume).
        save_only_model=on_tpu,
        bf16=settings["bf16"],
        gradient_checkpointing=not settings["no_gradient_checkpointing"],
        # Streaming HF datasets are I/O-bound on the main thread; workers > 0
        # gives no throughput but causes segfaults on small-/dev/shm containers
        # (e.g. JupyterHub) and tokenizer-fork races.
        dataloader_num_workers=0,
        # XLA requires static shapes — partial last batch triggers infinite
        # recompilation. Harmless on CUDA too.
        dataloader_drop_last=True,
        report_to=["wandb", "tensorboard"],
        max_grad_norm=1.0,
        # Streaming datasets have no __len__; required so Trainer doesn't try.
        ignore_data_skip=True,
        # Logs `train/num_input_tokens_seen` to WandB/TensorBoard.
        include_num_input_tokens_seen=True,
        **fsdp_kwargs,
    )

    # HF Trainer refuses `optimizers=` under FSDP. Optimizer creation order is
    # patched via TrainerWithPerplexity.create_scheduler override above.
    trainer = TrainerWithPerplexity(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    resume_from = find_latest_checkpoint(output_dir) if args.resume else None
    if args.resume:
        print(f"Resume requested; latest checkpoint: {resume_from or '(none — starting fresh)'}")

    trainer.train(resume_from_checkpoint=resume_from)
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved final model to {final_dir}")


if __name__ == "__main__":
    main()
