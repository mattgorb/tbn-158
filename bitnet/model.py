"""Build Llama-3-style models with optional BitLinear158 / TBNBitLinear158
replacement of every internal nn.Linear.

The four sizes mirror the configs that were in the torchtitan fork:
- 200M: 16 layers, hidden 768, GQA 12/4
- 500M: 28 layers, hidden 1024, GQA 16/4
- 700M: 24 layers, hidden 1536, GQA 24/8  (matches BitNet b1.58 paper)
- 3B:   28 layers, hidden 3072, GQA 24/8

Embedding (input) and lm_head stay full precision per the b1.58 paper.
"""

from functools import partial

import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from .linears import BitLinear158, TBNBitLinear158


VOCAB_SIZE = 128256  # Llama 3 tokenizer vocab
DEFAULT_MAX_POSITION = 4096

SIZE_HPARAMS = {
    "200M": dict(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,
        num_hidden_layers=16,
        intermediate_size=2048,
    ),
    "500M": dict(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_hidden_layers=28,
        intermediate_size=2816,
    ),
    "700M": dict(
        hidden_size=1536,
        num_attention_heads=24,
        num_key_value_heads=8,
        num_hidden_layers=24,
        intermediate_size=4096,
    ),
    "3B": dict(
        hidden_size=3072,
        num_attention_heads=24,
        num_key_value_heads=8,
        num_hidden_layers=28,
        intermediate_size=8192,
    ),
}


def build_llama_config(size: str, max_position_embeddings: int = DEFAULT_MAX_POSITION) -> LlamaConfig:
    if size not in SIZE_HPARAMS:
        raise ValueError(f"Unknown size {size!r}; choose from {list(SIZE_HPARAMS)}")
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        attention_bias=False,
        mlp_bias=False,
        **SIZE_HPARAMS[size],
    )


def _replace_linears(module: nn.Module, linear_factory, skip_module_names: set) -> int:
    """Recursively replace every ``nn.Linear`` in ``module`` with the result of
    ``linear_factory(in_features, out_features, bias)``, skipping any direct
    child whose attribute name is in ``skip_module_names``.

    Returns the number of layers replaced.
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name not in skip_module_names:
            new_linear = linear_factory(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
            )
            # Preserve initialization (HF inits at construction; otherwise the
            # ternary weight would start from a fresh kaiming-uniform).
            with_init_state = new_linear
            with_init_state.weight.data.copy_(child.weight.data)
            if child.bias is not None and new_linear.bias is not None:
                with_init_state.bias.data.copy_(child.bias.data)
            setattr(module, name, new_linear)
            replaced += 1
        else:
            replaced += _replace_linears(child, linear_factory, skip_module_names)
    return replaced


LINEAR_FACTORIES = {
    "plain": nn.Linear,
    "bitnet158": BitLinear158,
    "tbn158": partial(TBNBitLinear158, tile_size=2),
}


def build_model(size: str, variant: str) -> LlamaForCausalLM:
    """Instantiate a fresh (randomly initialized) Llama-style model at the
    requested size and replace internal nn.Linear with the chosen variant.
    """
    if variant not in LINEAR_FACTORIES:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {list(LINEAR_FACTORIES)}"
        )
    config = build_llama_config(size)
    model = LlamaForCausalLM(config)
    if variant != "plain":
        # ``lm_head`` is a top-level nn.Linear we keep full precision; the
        # input embedding is an nn.Embedding, so isinstance filtering already
        # skips it.
        _replace_linears(
            model,
            LINEAR_FACTORIES[variant],
            skip_module_names={"lm_head"},
        )
    return model
