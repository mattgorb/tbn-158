"""Pretrain a Llama-3-style model with optional BitNet b1.58 / TBN b1.58
linears, on streaming C4. Single-GPU friendly; uses HuggingFace Trainer.

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
import os
from itertools import chain
from pathlib import Path

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
    p.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable gradient checkpointing (faster, more memory).",
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


def build_dataset(tokenizer, seq_len: int):
    """Stream C4, tokenize, and pack into fixed seq_len blocks."""
    raw = load_dataset("allenai/c4", "en", split="train", streaming=True)

    def tokenize(batch):
        return tokenizer(batch["text"])

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )

    def group_into_blocks(batch):
        concatenated = {k: list(chain(*batch[k])) for k in batch}
        total = (len(concatenated["input_ids"]) // seq_len) * seq_len
        result = {
            k: [v[i : i + seq_len] for i in range(0, total, seq_len)]
            for k, v in concatenated.items()
        }
        result["labels"] = [list(ids) for ids in result["input_ids"]]
        return result

    return tokenized.map(group_into_blocks, batched=True)


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

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Streaming C4 (allenai/c4, en split)")
    train_ds = build_dataset(tokenizer, settings["seq_len"])

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=settings["per_device_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        max_steps=settings["steps"],
        warmup_steps=settings["warmup_steps"],
        learning_rate=settings["lr"],
        weight_decay=settings["weight_decay"],
        lr_scheduler_type="cosine",
        logging_steps=settings["logging_steps"],
        save_steps=settings["save_steps"],
        save_total_limit=settings["save_total_limit"],
        bf16=True,
        gradient_checkpointing=not settings["no_gradient_checkpointing"],
        dataloader_num_workers=2,
        report_to=["wandb", "tensorboard"],
        max_grad_norm=1.0,
        # Streaming datasets have no __len__; required so Trainer doesn't try.
        ignore_data_skip=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
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
