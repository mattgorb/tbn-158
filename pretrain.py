"""Pretrain a Llama-3-style model with optional BitNet b1.58 / TBN b1.58
linears, on streaming C4. Single-GPU friendly; uses HuggingFace Trainer.

Usage
-----
    # Smallest model, TBN-tiled b1.58:
    python pretrain.py --size 200M --variant tbn158

    # 700M plain baseline, override training length:
    python pretrain.py --size 700M --variant plain --steps 50000

    # Resume from the latest checkpoint in --output_dir:
    python pretrain.py --size 200M --variant tbn158 --resume

All four sizes (200M / 500M / 700M / 3B) and three variants (plain /
bitnet158 / tbn158) are supported. Per-size batch and learning-rate defaults
target a single 48 GB GPU (A6000); override with CLI flags.
"""

import argparse
import os
from itertools import chain
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from bitnet import LINEAR_FACTORIES, SIZE_HPARAMS, build_model


# Per-size defaults tuned for a single 48 GB GPU. Override with CLI flags.
SIZE_DEFAULTS = {
    "200M": dict(per_device_batch_size=16, lr=5e-4),
    "500M": dict(per_device_batch_size=8, lr=4e-4),
    "700M": dict(per_device_batch_size=4, lr=4e-4),
    "3B": dict(per_device_batch_size=1, lr=3e-4),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--size",
        choices=list(SIZE_HPARAMS),
        required=True,
        help="Model size.",
    )
    p.add_argument(
        "--variant",
        choices=list(LINEAR_FACTORIES),
        required=True,
        help="Linear flavor used inside attention/FFN.",
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
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--per_device_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--wandb_project", default="bitnet158")
    p.add_argument("--run_name", default=None)
    p.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (faster, more memory).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from the latest checkpoint in --output_dir.",
    )
    return p.parse_args()


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

    defaults = SIZE_DEFAULTS[args.size]
    if args.per_device_batch_size is None:
        args.per_device_batch_size = defaults["per_device_batch_size"]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.output_dir is None:
        args.output_dir = f"./outputs/{args.size}_{args.variant}"
    if args.run_name is None:
        args.run_name = f"{args.size}_{args.variant}"

    print(f"Building model: {args.size} / {args.variant}")
    model = build_model(args.size, args.variant)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {num_params:,}")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Streaming C4 (allenai/c4, en split)")
    train_ds = build_dataset(tokenizer, args.seq_len)

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
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

    resume_from = find_latest_checkpoint(args.output_dir) if args.resume else None
    if args.resume:
        print(f"Resume requested; latest checkpoint: {resume_from or '(none — starting fresh)'}")

    trainer.train(resume_from_checkpoint=resume_from)
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"Saved final model to {final_dir}")


if __name__ == "__main__":
    main()
