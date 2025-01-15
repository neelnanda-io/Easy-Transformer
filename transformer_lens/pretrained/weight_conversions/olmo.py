import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_olmo_weights(olmo, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.d_mlp is not None

    state_dict["embed.W_E"] = olmo.model.embed_tokens.weight
    for l in range(cfg.n_layers):
        olmo_layer = olmo.model.layers[l]

        W_Q = olmo_layer.self_attn.q_proj.weight
        W_K = olmo_layer.self_attn.k_proj.weight
        W_V = olmo_layer.self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        W_O = olmo_layer.self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_in"] = olmo_layer.mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = olmo_layer.mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.mlp.W_out"] = olmo_layer.mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln1.w"] = torch.ones(cfg.d_model, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.ln1.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.ln2.w"] = torch.ones(cfg.d_model, dtype=cfg.dtype)
        state_dict[f"blocks.{l}.ln2.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["ln_final.w"] = torch.ones(cfg.d_model, dtype=cfg.dtype)
    state_dict["ln_final.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

    state_dict["unembed.W_U"] = olmo.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
