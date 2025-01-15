from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.factories.activation_function_factory import (
    ActivationFunctionFactory,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class MoEGatedMLP(nn.Module):
    """MoEGated MLP

    This MLP matches the implementation for Mixtral on HuggingFace. It is meant to stay within our
    MoE, since the format of this MLP is different from the standard MLPs throughout
    TransformerLens.

    It may be possible to rework this to follow the same interface as other MLPs, but for the
    time being it is being left as is to ensure accuracy.
    """

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.d_mlp = self.cfg.d_mlp

        if self.d_mlp is None:
            raise ValueError("d_mlp must be set to use an MLP")

        self.W_in = nn.Linear(self.cfg.d_model, self.d_mlp, bias=False)
        self.W_out = nn.Linear(self.d_mlp, self.cfg.d_model, bias=False)
        self.W_gate = nn.Linear(self.cfg.d_model, self.d_mlp, bias=False)

        # hook on gate output but before act_fn
        self.hook_gate = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        self.act_fn = ActivationFunctionFactory.pick_activation_function(self.cfg)

    def forward(self, x: Float[torch.Tensor, "pos d_model"]) -> Float[torch.Tensor, "pos d_model"]:
        gated_x = self.hook_gate(self.W_gate(x))
        pre_act = self.hook_pre(self.W_in(x))
        post_act = self.hook_post(self.act_fn(gated_x) * pre_act)
        return self.W_out(post_act)


class MoE(CanBeUsedAsMLP):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__(cfg)

        # Ensure that num_experts and experts_per_token are specified and non-zero
        assert self.cfg.num_experts is not None, "num_experts must be specified for MoE layer"
        assert self.cfg.experts_per_token, "experts_per_token must be specified for MoE layer"

        self.num_experts: int = self.cfg.num_experts
        self.experts_per_token: int = self.cfg.experts_per_token

        assert (
            self.cfg.experts_per_token <= self.cfg.num_experts
        ), "experts_per_token must be less than or equal to num_experts"

        self.experts = nn.ModuleList([MoEGatedMLP(self.cfg) for _ in range(self.num_experts)])
        self.W_gate = nn.Linear(self.cfg.d_model, self.cfg.num_experts, bias=False)

        # Hook on the weights of selected experts [batch pos experts_per_token]
        self.hook_expert_weights = HookPoint()
        # Hook on the indices of selected experts [batch pos experts_per_token]
        self.hook_expert_indices = HookPoint()

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # [batch, pos, d_model] -> [batch, pos, num_experts]
        batch, pos, d_model = x.shape
        x = x.view(-1, d_model)
        gate_logits = self.W_gate(x)

        # choose the top k(=experts_per_token) experts to use
        # both are [batch, pos, experts_per_token]
        weights = self.hook_expert_weights(F.softmax(gate_logits, dim=1, dtype=torch.float))
        weights, expert_indices = torch.topk(weights, self.experts_per_token, dim=-1)
        if self.cfg.norm_topk_prob:
            weights /= weights.sum(dim=-1, keepdim=True)
        expert_indices = self.hook_expert_indices(expert_indices)
        weights = weights.to(x.dtype)

        results = torch.zeros((batch * pos, d_model), dtype=x.dtype, device=x.device)
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x].reshape(-1, d_model)

            current_hidden_states = expert_layer(current_state) * weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            results.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        results = results.reshape(batch, pos, d_model)
        return results
