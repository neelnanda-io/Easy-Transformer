"""Hooked Encoder Bert NSP Head Component.

This module contains all the component :class:`BertNSPHead`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class BertNSPHead(nn.Module):
    """
    Transforms BERT embeddings into logits. The purpose of this module is to predict whether or not sentence B follows sentence A.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W = nn.Parameter(torch.empty(self.cfg.d_model, 2, dtype=self.cfg.dtype))
        self.b = nn.Parameter(torch.zeros(2, dtype=self.cfg.dtype))
        self.hook_nsp_out = HookPoint()

    def forward(
        self, resid: Float[torch.Tensor, "batch d_model"]
    ) -> Float[torch.Tensor, "batch 2"]:
        nsp_logits = torch.matmul(resid, self.W) + self.b
        return self.hook_nsp_out(nsp_logits)
