import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_olmoe_weights(olmoe, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None
    assert cfg.d_mlp is not None
    assert cfg.num_experts is not None

    state_dict["embed.W_E"] = olmoe.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        olmoe_layer = olmoe.model.layers[l]
        state_dict[f"blocks.{l}.ln1.w"] = olmoe_layer.input_layernorm.weight

        W_Q = olmoe_layer.self_attn.q_proj.weight
        W_K = olmoe_layer.self_attn.k_proj.weight
        W_V = olmoe_layer.self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V
        state_dict[f"blocks.{l}.attn.q_norm.w"] = olmoe_layer.self_attn.q_norm.weight
        state_dict[f"blocks.{l}.attn.k_norm.w"] = olmoe_layer.self_attn.k_norm.weight

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
        )

        W_O = olmoe_layer.self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = olmoe_layer.post_attention_layernorm.weight

        state_dict[f"blocks.{l}.mlp.W_gate.weight"] = olmoe_layer.mlp.gate.weight

        for e in range(cfg.num_experts):
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = olmoe_layer.mlp.experts[
                e
            ].up_proj.weight
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = olmoe_layer.mlp.experts[
                e
            ].gate_proj.weight
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = olmoe_layer.mlp.experts[
                e
            ].down_proj.weight

    state_dict["ln_final.w"] = olmoe.model.norm.weight

    state_dict["unembed.W_U"] = olmoe.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
