"""Hooked Transformer Bert Embed Component.

This module contains all the component :class:`BertEmbed`.
"""
from typing import Dict, Optional, Union

import einops
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import Embed, LayerNorm, PosEmbed, TokenTypeEmbed
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class BertEmbed(nn.Module):
    """
    Custom embedding layer for a BERT-like model. This module computes the sum of the token, positional and token-type embeddings and takes the layer norm of the result.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.embed = Embed(self.cfg)
        self.pos_embed = PosEmbed(self.cfg)
        self.token_type_embed = TokenTypeEmbed(self.cfg)
        self.ln = LayerNorm(self.cfg)

        self.hook_embed = HookPoint()
        self.hook_pos_embed = HookPoint()
        self.hook_token_type_embed = HookPoint()

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch pos"],
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        base_index_id = torch.arange(input_ids.shape[1], device=input_ids.device)
        index_ids = einops.repeat(base_index_id, "pos -> batch pos", batch=input_ids.shape[0])
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings_out = self.hook_embed(self.embed(input_ids))
        position_embeddings_out = self.hook_pos_embed(self.pos_embed(index_ids))
        token_type_embeddings_out = self.hook_token_type_embed(
            self.token_type_embed(token_type_ids)
        )

        embeddings_out = word_embeddings_out + position_embeddings_out + token_type_embeddings_out
        layer_norm_out = self.ln(embeddings_out)
        return layer_norm_out
