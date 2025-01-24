"""Hooked Transformer Config.

Module with a dataclass for storing the configuration of a
:class:`transformer_lens.HookedTransformer` model.
"""

from __future__ import annotations

import logging
import pprint
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from transformer_lens import utils
from transformer_lens.utilities.activation_functions import SUPPORTED_ACTIVATIONS


@dataclass
class HookedTransformerConfig:
    """
    Configuration class to store the configuration of a HookedTransformer model.

    See further_comments.md for more details on the more complex arguments.

    Args:
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_layers (int): The number of transformer blocks (one block = one attn layer AND one MLP layer).
        n_ctx (int): The maximum sequence length.
        n_heads (int): The number of attention heads. If not
            specified, will be set to d_model // d_head. (This is represented by a default value of -1)
        d_mlp (int, *optional*): The dimensionality of the feedforward mlp
            network. Defaults to 4 * d_model, and in an attn-only model is None.
        d_vocab (int): The size of the vocabulary. Defaults to -1, which means not set. If not set, will be
            automatically set from the tokenizer's vocab size.
        act_fn (str, *optional*): The activation function to use. Always
            lowercase. Supports ['relu', 'gelu', 'silu', 'gelu_new', 'solu_ln',
            'gelu_fast']. Must be set unless using an attn-only model.
        eps (float): The epsilon value to use for layer normalization. Defaults
            to 1e-5
        use_attn_result (bool): whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        use_split_qkv_input (bool): whether to explicitly calculate the input of
            each head separately, with a hook. Defaults to false to save memory.
        use_hook_mlp_in (bool): whether to use a hook to get the input to the
            MLP layer. Defaults to false to save memory.
        use_attn_in (bool): whether to explicitly calculate the input of each
            attention head separately, with a hook. Defaults to false to save memory
        use_attn_scale (bool): whether to scale the attention weights by
            1/sqrt(d_head)
        ungroup_grouped_query_attention (bool): whether to ungroup key and value heads, for models that use
            grouped query attention.
        attn_scale (float): The amount to divide attention scores by (if applicable). Defaults to
            sqrt(d_head)
        model_name (str): the name of the model, used to load
            weights from HuggingFace or initialized to "custom" if not passed
        original_architecture (str, *optional*): the family of the model, used
        to help load
            weights from HuggingFace or initialized to "custom" if not passed
        from_checkpoint (bool): Whether the model weights were
            loaded from a checkpoint (only applies to pretrained models)
        checkpoint_index (int, *optional*): The index of the
            checkpoint loaded (only applies to pretrained models).
        checkpoint_label_type (str, *optional*): Whether
            checkpoints are labelled by the number of steps or number of tokens.
        checkpoint_value (int, *optional*): The value of the
            checkpoint label (whether of steps or tokens).
        tokenizer_name (str, *optional*): the full name of the model, passed into
            HuggingFace to access the tokenizer. Only used when passing in
            custom config, if loading from pretrained then this is not needed.
        use_local_attn (bool): whether to use local attention - ie each
            destination token can only attend to source tokens a certain distance back.
        window_size (int, *optional*): the size of the window for local
            attention
        attn_types (List[str], *optional*): the types of attention to use for
            local attention
        init_mode (str): the initialization mode to use for the
            weights. Only relevant for custom models, ignored for pre-trained.
            We now support 'gpt2', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
            'kaiming_normal'. MuP support to come. Defaults to 'gpt2'.
        normalization_type (str, *optional*): the type of normalization to use.
            Options are None (no normalization), 'LN' (use LayerNorm, including weights
            & biases) and 'LNPre' (use LayerNorm, but no weights or biases), 'RMS'
            (use RMSNorm, including weights) and 'RMSPre' (use RMSNorm, but no weights or biases).
            Defaults to LN
        device(str): The device to use for the model. Defaults to 'cuda' if
            available, else 'cpu'. Must be 'cuda' if `n_devices` > 1.
        n_devices (int): The number of devices to use for the model. Defaults to 1. Layers are loaded
            to support "pipeline parallelism", where each device is responsible for a subset of the layers.
        attention_dir (str): Whether to use causal (aka unidirectional aka GPT-2
            style) or bidirectional attention. Options are 'causal' and
            'bidirectional'. Defaults to 'causal'
        attn_only (bool): Whether to only use attention layers, no feedforward
            layers. Defaults to False
        seed (int, *optional*): The seed to use for the model.
            Used to set sources of randomness (Python, PyTorch and NumPy) and to initialize weights.
            Defaults to None. We recommend setting a seed, so your experiments are reproducible.
        initializer_range (float): The standard deviation of the normal used to
            initialise the weights, initialized to 0.8 / sqrt(d_model). If init_mode is
            'xavier_uniform' or 'xavier_normal', this value is instead treated as the `gain` parameter for the weight
            initialisation (a constant factor to scale the weights by). Defaults to -1.0, which means not set.
        init_weights (bool): Whether to initialize the weights. Defaults to
            True. If False, does not initialize weights.
        scale_attn_by_inverse_layer_idx (bool): Whether to scale the attention
            weights by 1/(layer_id+1), used by Mistral (Stanford) models for numerical stability when
            training in FP16. Defaults to False.
        positional_embedding_type (str): The positional embedding used. Options
            are 'standard' (ie GPT-2 style, absolute, randomly initialized learned positional
            embeddings, directly added to the residual stream), 'rotary'
            (described here: https://blog.eleuther.ai/rotary-embeddings/ ) and
            'shortformer' (GPT-2 style absolute & learned, but rather than being
            added to the residual stream they're only added to the inputs to the
            keys and the queries (ie key = W_K(res_stream + pos_embed), but
            values and MLPs don't get any positional info)). Sinusoidal are not
            currently supported. Defaults to 'standard'.
        final_rms (bool): Whether to replace the final normalization (just
            before the unembed) with RMSNorm (ie no centering or bias, just
            scaling + weights). Only included because of a dumb bug in my
            original SoLU code. Defaults to False.
        d_vocab_out (int, *optional*): The size of the output vocabulary. Defaults to -1, which means not set. If not
            set, will be equal to d_vocab. Mainly useful for algorithmic tasks
            where the input and output vocabularies may be different.
        parallel_attn_mlp (bool): Whether to parallelize the attention and MLP
            layers - a weird cursed thing done by GPT-J. Means that
            mlp_out=MLP(ln1(resid_pre)) and resid_post=resid_pre+attn_out+mlp_out. Defaults to False.
        rotary_dim (int, *optional*): The dimensionality of the rotary
            embeddings, may be d_head in which case only the first rotary_dim
            dimensions of each head are rotated. Defaults to None, if
            positional_embedding_type=="rotary" post-init then sets it to d_head, i.e. "rotate all
            dimensions of the query and key".
        n_params (int, *optional*): The number of (hidden weight)
            parameters in the model. This is automatically calculated and not
            intended to be set by the user. (Non embedding parameters, because
            the [scaling laws paper](https://arxiv.org/pdf/2001.08361.pdf) found
            that that was a more meaningful number. Ignoring biases and layer
            norms, for convenience)
        use_hook_tokens (bool): Will add a hook point on the token input to
            HookedTransformer.forward, which lets you cache or intervene on the tokens.
            Defaults to False.
        default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
            methods of HookedTransformer process input text to tokenize (only when input is a string).
            Defaults to True - even for models not explicitly trained with this, heads often use the
            first position as a resting position and accordingly lose information from the first token,
            so this empirically seems to give better results. To change the default behavior to False, pass in
            default_prepend_bos=False. Note that you can also locally override the default behavior by passing
            in prepend_bos=True/False when you call a method that processes the input string.
        dtype (torch.dtype, *optional*): The model's dtype. Defaults to torch.float32.
        tokenizer_prepends_bos (bool, *optional*): This flag is set by set_tokenizer. It is set to True only
            when the tokenizer automatically prepends the BOS token if initialized with add_bos_token=True.
            We need this information to dynamically control bos prepending.
        load_in_4bit(bool): If this flag is set, then it's assumed that parameters are 4-bit quantized
            with bitsandbytes. Currently only supported for Llama.
        n_key_value_heads (int, *optional*): The number of groups of heads that use the same key and value matrix.
            Only for models that use Grouped Query Attention.
        post_embedding_ln (bool): Whether to apply layer normalization after embedding the tokens. Defaults
            to False.
        num_experts (int, *optional*): The number of experts to use in the MoE layer. If set, experts_per_token
            must also be set. Set to None if not using MoE.
        experts_per_token (int, *optional*): The number of experts to use for each pass in the MoE layer. If set,
            num_experts must also be set. Set to None if not using MoE.
        relative_attention_max_distance (int, *optional*): The maximum distance between tokens for relative
            attention. If set, relative_attention_num_buckets must also be set.Only used in EncoderDecoder models, like T5.
        relative_attention_num_buckets (int, *optional*): The number of buckets to use for relative attention.
            If set, relative_attention_max_distance must also be set.Only used in EncoderDecoder models, like T5.
        decoder_start_token_id (int, *optional*): The start token id for the decoder. Only used in EncoderDecoder models, like T5.
        tie_word_embeddings (bool): Whether to tie the word embeddings and the output layer weights. Defaults to False. Only used in EncoderDecoder (T5) by now.
        use_normalization_before_and_after (bool): Whether to apply normalization (LN/RMS/etc)
            to both the input of an attn/MLP block *and* the output (before adding back to the
            residual stream). Currently only used in Gemma-2. Defaults to False.
        attn_scores_soft_cap (float): An optional softcap for attention scores pre-softmax. If
            used, it will map attn_scores -> soft_cap * tanh(attn_scores / soft_cap). As tanh's
            output is in [-1, 1], this maps attn_scores to [-soft_cap, soft_cap], with little
            effect on small values, but squashing large values into that interval. Currently only
            used in Gemma-2. Defaults to -1.0, which means not set.
        output_logits_soft_cap (float): An optional softcap for output logits, currently only used
            in Gemma-2 (see attn_scores_soft_cap for details). Defaults to -1.0, which means not
            set.
        use_NTK_by_parts_rope (bool): Whether to apply the "NTK-by-parts" method when using Rotary
            Positional Embedding. This method adjusts the interpolation based on frequency factors
            for different parts of the hidden dimensions. See Section 3.2 in
            https://arxiv.org/pdf/2309.00071 for details. Defaults to False.
        NTK_by_parts_low_freq_factor (float): The threshold applied to low-frequency hidden
            dimensions during interpolation when using the "NTK-by-parts" method. Defaults to 1.0.
        NTK_by_parts_high_freq_factor (float): The threshold applied to high-frequency hidden
            dimensions during interpolation in the "NTK-by-parts" method. Defaults to 4.0.
        NTK_by_parts_factor (float): The overall factor used in the "NTK-by-parts" method that
            affects the rate of change between low and high-frequency interpolation strategies.
            Defaults to 8.0.


    """

    n_layers: int
    d_model: int
    n_ctx: int
    d_head: int
    model_name: str = "custom"
    n_heads: int = -1
    d_mlp: Optional[int] = None
    act_fn: Optional[str] = None
    d_vocab: int = -1
    eps: float = 1e-5
    use_attn_result: bool = False
    use_attn_scale: bool = True
    attn_scale: float = -1.0
    use_split_qkv_input: bool = False
    use_hook_mlp_in: bool = False
    use_attn_in: bool = False
    use_local_attn: bool = False
    ungroup_grouped_query_attention: bool = False
    original_architecture: Optional[str] = None
    from_checkpoint: bool = False
    checkpoint_index: Optional[int] = None
    checkpoint_label_type: Optional[str] = None
    checkpoint_value: Optional[int] = None
    tokenizer_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None
    init_mode: str = "gpt2"
    normalization_type: Optional[str] = "LN"
    device: Optional[str] = None
    n_devices: int = 1
    attention_dir: str = "causal"
    attn_only: bool = False
    seed: Optional[int] = None
    initializer_range: float = -1.0
    init_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    positional_embedding_type: str = "standard"
    final_rms: bool = False
    d_vocab_out: int = -1
    parallel_attn_mlp: bool = False
    rotary_dim: Optional[int] = None
    n_params: Optional[int] = None
    use_hook_tokens: bool = False
    gated_mlp: bool = False
    default_prepend_bos: bool = True
    dtype: torch.dtype = torch.float32
    tokenizer_prepends_bos: Optional[bool] = None
    n_key_value_heads: Optional[int] = None
    post_embedding_ln: bool = False
    rotary_base: float = 10000.0
    trust_remote_code: bool = False
    rotary_adjacent_pairs: bool = False
    load_in_4bit: bool = False
    num_experts: Optional[int] = None
    experts_per_token: Optional[int] = None
    relative_attention_max_distance: Optional[int] = None
    relative_attention_num_buckets: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    tie_word_embeddings: bool = False
    use_normalization_before_and_after: bool = False
    attn_scores_soft_cap: float = -1.0
    output_logits_soft_cap: float = -1.0
    use_NTK_by_parts_rope: bool = False
    NTK_by_parts_low_freq_factor: float = 1.0
    NTK_by_parts_high_freq_factor: float = 4.0
    NTK_by_parts_factor: float = 8.0

    def __post_init__(self):
        if self.n_heads == -1:
            self.n_heads = self.d_model // self.d_head

            if not self.d_model % (self.d_head) == 0:
                logging.warning(
                    "d_model %d is not divisible by d_head %d."
                    "n_heads was inferred to be %d, rounding down the ratio.",
                    self.d_model,
                    self.d_head,
                    self.n_heads,
                )

        if self.seed is not None:
            self.set_seed_everywhere(self.seed)
        if self.use_local_attn:
            assert self.window_size is not None, "window_size must be specified for local attention"
            assert self.attn_types is not None, "attn_types must be specified for local attention"
        if not self.attn_only:
            if self.d_mlp is None:
                # For some reason everyone hard codes in this hyper-parameter!
                self.d_mlp: int = self.d_model * 4
            assert self.act_fn is not None, "act_fn must be specified for non-attn-only models"
            assert (
                self.act_fn in SUPPORTED_ACTIVATIONS
            ), f"act_fn={self.act_fn} must be one of {SUPPORTED_ACTIVATIONS}"
        if self.initializer_range < 0 and self.init_mode == "gpt2":
            # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
            self.initializer_range = 0.8 / np.sqrt(self.d_model)
        if self.initializer_range < 0 and self.init_mode != "gpt2":
            # This is the gain parameter for the weight initialisation
            self.initializer_range = 1.0

        if self.d_vocab_out == -1:
            # d_vocab_out defaults to d_vocab, unless there's an algorithmic task
            # If d_vocab is not set, it'll be inferred from tokenizer_name or from a tokenizer
            # explicitly passed to HookedTransformer initialisation.
            self.d_vocab_out = self.d_vocab

        if self.positional_embedding_type == "rotary" and self.rotary_dim is None:
            self.rotary_dim = self.d_head

        if self.num_experts is not None:
            assert (
                self.experts_per_token is not None
            ), "experts_per_token must be set if num_experts is set"
        if self.experts_per_token is not None:
            assert (
                self.num_experts is not None
            ), "num_experts must be set if experts_per_token is set"

        # The number of parameters in attention layers (ignoring biases and layer norm). 4 because W_Q, W_K, W_V and W_O
        self.n_params = self.n_layers * ((self.d_model * self.d_head * self.n_heads * 4))
        if not self.attn_only:
            assert self.d_mlp is not None  # mypy
            # Number of parameters in MLP layers (ignoring biases and layer norm). 2 because W_in and W_out
            mlp_params_per_layer = self.d_model * self.d_mlp * (2 + self.gated_mlp)

            if self.num_experts:
                # If we are using MoE, we multiply by num_experts, and add the expert gate parameters (d_model * num_experts)
                mlp_params_per_layer = (mlp_params_per_layer + self.d_model) * self.num_experts
            self.n_params += self.n_layers * mlp_params_per_layer

        if self.device is None:
            self.device = utils.get_device()

        if self.n_devices > 1:
            assert (
                torch.cuda.device_count() >= self.n_devices
            ), f"Not enough CUDA devices to support n_devices {self.n_devices}"

        if self.use_attn_scale and self.attn_scale == -1.0:
            self.attn_scale = np.sqrt(self.d_head)

        assert self.default_prepend_bos in [
            True,
            False,
        ], f"padding_side must be either True or False, but {self.default_prepend_bos} is given"

    @classmethod
    def unwrap(cls, config: Union[Dict, "HookedTransformerConfig"]) -> HookedTransformerConfig:
        """
        Convenience function to avoid duplicate code from a common way config is passed to various components
        """
        return HookedTransformerConfig.from_dict(config) if isinstance(config, Dict) else config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HookedTransformerConfig:
        """
        Instantiates a `HookedTransformerConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedTransformerConfig:\n" + pprint.pformat(self.to_dict())

    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def is_layer_norm_activation(self) -> bool:
        return self.act_fn is not None and self.act_fn.endswith("_ln")
