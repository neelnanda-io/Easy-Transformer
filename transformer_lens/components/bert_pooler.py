"""Hooked Encoder Bert Pooler Component.

This module contains all the component :class:`BertPooler`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class BertPooler(nn.Module):
    """
    Transforms the [CLS] token representation into a fixed-size sequence embedding.
    The purpose of this module is to convert variable-length sequence inputs into a single vector representation suitable for downstream tasks.
    (e.g. Next Sentence Prediction)
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_model, dtype=self.cfg.dtype))
        self.b = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))
        self.activation = nn.Tanh()
        self.hook_pooler_out = HookPoint()

    def forward(
        self, resid: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch d_model"]:
        first_token_tensor = resid[:, 0]
        pooled_output = torch.matmul(first_token_tensor, self.W) + self.b
        pooled_output = self.hook_pooler_out(self.activation(pooled_output))
        return pooled_output
