# BitNet b1.58 (arXiv:2402.17764) and a TBN-tiled b1.58 variant for torchtitan.
#
# Drop-in replacements for `torchtitan.models.common.linear.Linear`. To use them, swap
# `Linear.Config(...)` for `BitLinear158.Config(...)` or `TBNBitLinear158.Config(...)` in any
# attention / feed-forward config builder. Both classes inherit from `nn.Linear` so FSDP / param
# init / state_dict all work unchanged; only `forward` is overridden.
#
# Two classes, both fully self-contained. Each contains its own activation and weight quantization
# logic inline — there are deliberately no shared helpers, so each class reads top-to-bottom.
#
# Training notes
# --------------
# * Underlying float weights are trained as usual. The forward sees a quantized version of the
#   weight; gradients flow to the float master via a straight-through estimator
#   (PyTorch idiom: `x + (x_q - x).detach()`).
# * Activations are quantized per-token with 8-bit absmax. No bias is supported (matches the paper
#   and standard Llama configs).
# * The b1.58 paper places a LayerNorm *before* every BitLinear input. Llama-style models already
#   apply RMSNorm before each attention / feed-forward block, so we do not add an extra norm here.
#
# Tensor-parallel caveat
# ----------------------
# Both classes compute `gamma = mean(|W|)` over the entire weight tensor. With FSDP this is fine
# (FSDP all-gathers the full weight before forward). With tensor parallelism the weight is sharded
# and the local mean is wrong; an allreduce across the TP group would be required. For 3B-scale
# pretraining with FSDP-only that is not an issue. Add `dist.all_reduce(gamma, op=AVG, group=tp)`
# in the forward if you ever turn TP on.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.module import Module


class BitLinear158(nn.Linear, Module):
    """BitNet b1.58: ternary {-1, 0, +1} weights via absmean rounding, 8-bit per-token absmax
    activations, straight-through estimator on backward.

    Drop-in replacement for `Linear`; same Config interface.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(config.in_features, config.out_features, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8-bit per-token absmax activation quantization with straight-through estimator.
        qb = 127.0
        gamma_x = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        x_q = (x * (qb / gamma_x)).round().clamp(-qb, qb) * (gamma_x / qb)
        x_used = x + (x_q - x).detach()

        # Ternary weight quantization: clip(round(W / gamma), -1, +1) * gamma, where gamma is the
        # per-tensor mean of |W| (b1.58's "absmean" rule). STE pushes gradients to the float master.
        w = self.weight
        gamma_w = w.abs().mean().clamp_min(1e-5)
        w_q = (w / gamma_w).round().clamp_(-1.0, 1.0) * gamma_w
        w_used = w + (w_q - w).detach()

        return F.linear(x_used, w_used, self.bias)


class TBNBitLinear158(nn.Linear, Module):
    """BitNet b1.58 applied to a tile-averaged weight: average every `tile_size` adjacent input
    columns of W before ternary quantization, then expand the resulting (out, in/p) trit tensor
    back to (out, in) for the matmul. Stored model holds `out * in / tile_size` trits plus one
    fp32 scale per tensor, so the packed artifact is ~tile_size x smaller than plain BitLinear158.

    Underlying float weight shape is unchanged (out_features, in_features). The tiling and quant
    happen inside forward; backward uses the same STE.

    Drop-in replacement for `Linear`; same Config interface plus `tile_size`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        bias: bool = False
        tile_size: int = 2

    def __init__(self, config: Config):
        super().__init__(config.in_features, config.out_features, bias=config.bias)
        if config.in_features % config.tile_size != 0:
            raise ValueError(
                f"TBNBitLinear158 in_features={config.in_features} must be divisible by "
                f"tile_size={config.tile_size}"
            )
        self.tile_size = config.tile_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8-bit per-token absmax activation quantization with straight-through estimator.
        qb = 127.0
        gamma_x = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        x_q = (x * (qb / gamma_x)).round().clamp(-qb, qb) * (gamma_x / qb)
        x_used = x + (x_q - x).detach()

        # Tile-averaged ternary weight quantization. Group every `tile_size` adjacent input columns
        # of W into one averaged column, ternary-quantize the smaller (out, in/p) matrix, then
        # broadcast it back to full input width for the matmul. STE flows through to the float W.
        w = self.weight
        out_features, in_features = w.shape
        p = self.tile_size
        w_tiled = w.view(out_features, in_features // p, p).mean(dim=-1)
        gamma_w = w_tiled.abs().mean().clamp_min(1e-5)
        w_q_tiled = (w_tiled / gamma_w).round().clamp_(-1.0, 1.0) * gamma_w
        w_q = (
            w_q_tiled.unsqueeze(-1)
            .expand(out_features, in_features // p, p)
            .reshape(out_features, in_features)
        )
        w_used = w + (w_q - w).detach()

        return F.linear(x_used, w_used, self.bias)
