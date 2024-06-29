"""Hooked Transformer MLP Component.

This module contains all the component :class:`MLP`.
"""

from typing import Callable, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.addmm import batch_addmm
from transformer_lens.utils import ACTIVATION_FN_DICT


# MLP Layers
class MLP(nn.Module):
    act_fn: Callable[..., torch.Tensor]
    ln: nn.Module

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        assert self.cfg.d_mlp is not None  # TODO: should this not be optional?
        self.W_in = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=self.cfg.dtype)
        )
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=self.cfg.dtype))
        self.W_out = nn.Parameter(
            torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=self.cfg.dtype)
        )
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.act_fn in ACTIVATION_FN_DICT:
            self.act_fn = ACTIVATION_FN_DICT[self.cfg.act_fn]
        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

        if self.cfg.act_fn == "solu_ln":
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # This is equivalent to (roughly) W_in @ x + b_in. It's important to
        # use a fused addmm to ensure it matches the Huggingface implementation
        # exactly.
        pre_act = self.hook_pre(batch_addmm(self.b_in, self.W_in, x))  # [batch, pos, d_mlp]
        if self.cfg.act_fn is not None and not self.cfg.act_fn.endswith("_ln"):
            post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        return batch_addmm(self.b_out, self.W_out, post_act)
