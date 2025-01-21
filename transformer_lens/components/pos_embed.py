"""Hooked Transformer POS Embed Component.

This module contains all the component :class:`PosEmbed`.
"""
from typing import Dict, Optional, Union

import einops
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utils import get_offset_position_ids


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W_pos = nn.Parameter(
            torch.empty(self.cfg.n_ctx, self.cfg.d_model, dtype=self.cfg.dtype)
        )

    def forward(
        self,
        tokens: Int[torch.Tensor, "batch pos"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch new_pos d_model"]:
        """
        Forward pass for positional embeddings.

        Args:
            tokens (Int[torch.Tensor, "batch pos"]): Input tokens.
            past_kv_pos_offset (int, optional): The length of tokens in the past_kv_cache. Defaults to 0.
            attention_mask (Int[torch.Tensor, "batch pos"], optional): The attention mask for padded tokens.
                 Defaults to None.

        Returns:
            Float[torch.Tensor, "batch pos d_model"]: Absolute position embeddings.
        """
        tokens_length = tokens.size(-1)

        if attention_mask is None:
            pos_embed = self.W_pos[
                past_kv_pos_offset : tokens_length + past_kv_pos_offset, :
            ]  # [pos, d_model]
            batch_pos_embed = einops.repeat(
                pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
            )

        else:
            # Separated from the no padding case for computational efficiency
            # (this code is a bit slower than the code above)

            offset_position_ids = get_offset_position_ids(past_kv_pos_offset, attention_mask)
            pos_embed = self.W_pos[offset_position_ids]  # [batch, pos, d_model]

            # Set the position embeddings to 0 for pad tokens (this is an arbitrary choice)
            padding_mask = ~attention_mask.bool()  # [batch, tokens_length]
            offset_padding_mask = padding_mask[
                :, past_kv_pos_offset : tokens_length + past_kv_pos_offset
            ].unsqueeze(
                -1
            )  # [batch, pos, 1]
            batch_pos_embed = torch.where(offset_padding_mask, 0, pos_embed)

        return batch_pos_embed.clone()
