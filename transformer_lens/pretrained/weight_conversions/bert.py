import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_bert_weights(bert, cfg: HookedTransformerConfig):
    embeddings = bert.bert.embeddings
    state_dict = {
        "embed.embed.W_E": embeddings.word_embeddings.weight,
        "embed.pos_embed.W_pos": embeddings.position_embeddings.weight,
        "embed.token_type_embed.W_token_type": embeddings.token_type_embeddings.weight,
        "embed.ln.w": embeddings.LayerNorm.weight,
        "embed.ln.b": embeddings.LayerNorm.bias,
    }

    for l in range(cfg.n_layers):
        block = bert.bert.encoder.layer[l]
        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            block.attention.self.query.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            block.attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            block.attention.self.key.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            block.attention.self.key.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            block.attention.self.value.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            block.attention.self.value.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            block.attention.output.dense.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = block.attention.output.dense.bias
        state_dict[f"blocks.{l}.ln1.w"] = block.attention.output.LayerNorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = block.attention.output.LayerNorm.bias
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            block.intermediate.dense.weight, "mlp model -> model mlp"
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = block.intermediate.dense.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            block.output.dense.weight, "model mlp -> mlp model"
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = block.output.dense.bias
        state_dict[f"blocks.{l}.ln2.w"] = block.output.LayerNorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = block.output.LayerNorm.bias

    pooler = bert.bert.pooler
    state_dict["pooler.W"] = pooler.dense.weight.T
    state_dict["pooler.b"] = pooler.dense.bias

    mlm_head = bert.cls.predictions
    state_dict["mlm_head.W"] = mlm_head.transform.dense.weight.T
    state_dict["mlm_head.b"] = mlm_head.transform.dense.bias
    state_dict["mlm_head.ln.w"] = mlm_head.transform.LayerNorm.weight
    state_dict["mlm_head.ln.b"] = mlm_head.transform.LayerNorm.bias

    # The NSP head does not have an unembedding
    # so we are only using weights from the MLM head
    # Note: BERT uses tied embeddings
    state_dict["unembed.W_U"] = mlm_head.decoder.weight.T
    state_dict["unembed.b_U"] = mlm_head.decoder.bias

    nsp_head = bert.cls.seq_relationship
    state_dict["nsp_head.W"] = nsp_head.weight.T
    state_dict["nsp_head.b"] = nsp_head.bias

    return state_dict
