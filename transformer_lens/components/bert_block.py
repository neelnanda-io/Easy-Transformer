"""Hooked Transformer Bert Block Component.

This module contains all the component :class:`BertBlock`.
"""
from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import Attention, LayerNorm
from transformer_lens.factories.mlp_factory import MLPFactory
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utils import repeat_along_head_dimension


class BertBlock(nn.Module):
    """
    BERT Block. Similar to the TransformerBlock, except that the LayerNorms are applied after the attention and MLP, rather than before.
    """

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.attn = Attention(cfg)
        self.ln1 = LayerNorm(cfg)
        self.mlp = MLPFactory.create_mlp(self.cfg)
        self.ln2 = LayerNorm(cfg)

        self.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]
        self.hook_normalized_resid_post = HookPoint()  # [batch, pos, d_model]

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        resid_pre = self.hook_resid_pre(resid_pre)

        query_input = resid_pre
        key_input = resid_pre
        value_input = resid_pre

        if self.cfg.use_split_qkv_input:
            n_heads = self.cfg.n_heads
            query_input = self.hook_q_input(repeat_along_head_dimension(query_input, n_heads))
            key_input = self.hook_k_input(repeat_along_head_dimension(key_input, n_heads))
            value_input = self.hook_v_input(repeat_along_head_dimension(value_input, n_heads))

        attn_out = self.hook_attn_out(
            self.attn(
                query_input,
                key_input,
                value_input,
                additive_attention_mask=additive_attention_mask,
            )
        )
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)

        mlp_in = resid_mid if not self.cfg.use_hook_mlp_in else self.hook_mlp_in(resid_mid.clone())
        normalized_resid_mid = self.ln1(mlp_in)
        mlp_out = self.hook_mlp_out(self.mlp(normalized_resid_mid))
        resid_post = self.hook_resid_post(normalized_resid_mid + mlp_out)
        normalized_resid_post = self.hook_normalized_resid_post(self.ln2(resid_post))

        return normalized_resid_post
