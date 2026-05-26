"""BitNet b1.58 (arXiv:2402.17764) and TBN-tiled b1.58 linear layers.

Drop-in replacements for ``nn.Linear``. Both inherit from ``nn.Linear`` so
weight init, state_dict, parameter registration, and gradient handling all
work unchanged; only ``forward`` is overridden.

Training notes
--------------
* Underlying float weights are trained as usual. The forward sees a quantized
  version of the weight; gradients flow to the float master via a
  straight-through estimator (``x + (x_q - x).detach()``).
* Activations are quantized per-token with 8-bit absmax. No bias is supported
  (matches the paper and standard Llama configs).
* The b1.58 paper places a LayerNorm before every BitLinear input. Llama-style
  models already apply RMSNorm before each attention / feed-forward block, so
  no extra norm is added here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear158(nn.Linear):
    """BitNet b1.58: ternary {-1, 0, +1} weights via absmean rounding, 8-bit
    per-token absmax activations, straight-through estimator on backward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8-bit per-token absmax activation quantization with STE.
        qb = 127.0
        gamma_x = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        x_q = (x * (qb / gamma_x)).round().clamp(-qb, qb) * (gamma_x / qb)
        x_used = x + (x_q - x).detach()

        # Ternary weight quantization: clip(round(W / gamma), -1, +1) * gamma,
        # where gamma is the per-tensor mean of |W| (b1.58's "absmean" rule).
        w = self.weight
        gamma_w = w.abs().mean().clamp_min(1e-5)
        w_q = (w / gamma_w).round().clamp(-1.0, 1.0) * gamma_w
        w_used = w + (w_q - w).detach()

        return F.linear(x_used, w_used, self.bias)


class TBNBitLinear158(nn.Linear):
    """BitNet b1.58 applied to a tile-averaged weight: average every
    ``tile_size`` adjacent input columns of W before ternary quantization,
    then broadcast the (out, in/p) trit tensor back to (out, in) for the
    matmul. Packed artifact is ~tile_size x smaller than plain BitLinear158.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        tile_size: int = 2,
    ):
        super().__init__(in_features, out_features, bias=bias)
        if in_features % tile_size != 0:
            raise ValueError(
                f"TBNBitLinear158 in_features={in_features} must be divisible "
                f"by tile_size={tile_size}"
            )
        self.tile_size = tile_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qb = 127.0
        gamma_x = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
        x_q = (x * (qb / gamma_x)).round().clamp(-qb, qb) * (gamma_x / qb)
        x_used = x + (x_q - x).detach()

        w = self.weight
        out_features, in_features = w.shape
        p = self.tile_size
        w_tiled = w.view(out_features, in_features // p, p).mean(dim=-1)
        gamma_w = w_tiled.abs().mean().clamp_min(1e-5)
        w_q_tiled = (w_tiled / gamma_w).round().clamp(-1.0, 1.0) * gamma_w
        w_q = (
            w_q_tiled.unsqueeze(-1)
            .expand(out_features, in_features // p, p)
            .reshape(out_features, in_features)
        )
        w_used = w + (w_q - w).detach()

        return F.linear(x_used, w_used, self.bias)
