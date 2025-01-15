"""Hooked Transformer.

The Hooked Transformer is the core part of TransformerLens.

In common PyTorch model implementations (e.g. ones from HuggingFace) it's fairly easy to extract
model weights, but much harder to extract activations. TransformerLens aims to simplify this task by
attaching hooks to every notable activation within the model. This enables the inspection and/or
alteration of activations in individual components like attention heads and MLP layers, facilitating
a deeper understanding of the internal workings of transformers like GPT-2.
"""

import logging
import os
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
from jaxtyping import Float, Int
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from typing_extensions import Literal

import transformer_lens.loading_from_pretrained as loading
import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import (
    Embed,
    LayerNorm,
    LayerNormPre,
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    TransformerBlock,
    Unembed,
)
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.loading_from_pretrained import NON_HF_HOSTED_MODEL_NAMES

# Note - activation cache is used with run_with_cache, past_key_value_caching is used for
# generation.
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from transformer_lens.utilities import devices
from transformer_lens.utils import (
    USE_DEFAULT_VALUE,
    init_kaiming_normal_,
    init_kaiming_uniform_,
    init_xavier_normal_,
    init_xavier_uniform_,
)

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

T = TypeVar("T", bound="HookedTransformer")


class Output(NamedTuple):
    """Output Named Tuple.

    Named tuple object for if we want to output both logits and loss.
    """

    logits: Float[torch.Tensor, "batch pos d_vocab"]
    loss: Loss


class HookedTransformer(HookedRootModule):
    """Hooked Transformer.

    Implements a full Transformer using the components :doc:`here <transformer_lens.components>`,
    with a :class:`transformer_lens.hook_points.HookPoint` on every interesting activation.

    TransformerLens comes loaded with >50 GPT-style models. Typically you initialise it with one of
    these via :meth:`from_pretrained`, although it can also be instantiated with randomly
    initialized weights via :meth:`__init__`.

    Once you've initialized the model, a common next step is to test it can do the task you're
    investigating. This can be done with :func:`transformer_lens.utils.test_prompt`.
    """

    ln_final: nn.Module

    def __init__(
        self,
        cfg: Union[HookedTransformerConfig, Dict],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        move_to_device: bool = True,
        default_padding_side: Literal["left", "right"] = "right",
    ):
        """Model initialization.

        Note that if you want to load the model from pretrained weights, you should use
        :meth:`from_pretrained` instead.

        Args:
            cfg: The config to use for the model.
            tokenizer: The tokenizer to use for the model. If not provided, it is inferred from
                `cfg.tokenizer_name` or initialized to `None`. If `None`, then the model cannot be
                passed strings, and d_vocab must be explicitly set.
            move_to_device: Whether to move the model to the device specified in cfg.
                device. Must be true if `n_devices` in the config is greater than 1, since the
                model's layers will be split across multiple devices.
            default_padding_side: Which side to pad on.
        """
        super().__init__()
        if isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a "
                "pretrained model, use HookedTransformer.from_pretrained() instead."
            )

        self.cfg = HookedTransformerConfig.unwrap(cfg)
        if tokenizer is not None:
            self.set_tokenizer(tokenizer, default_padding_side=default_padding_side)
        elif self.cfg.tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            if self.cfg.tokenizer_name in NON_HF_HOSTED_MODEL_NAMES:
                logging.warning(
                    "%s tokenizer not loaded. Please load manually.",
                    self.cfg.tokenizer_name,
                )
            else:
                # Hugging Face defaults to use_fast to True
                use_fast = True
                # Phi model's fast tokenizer does not support adding a BOS token, use_fast
                # should be False
                if "phi" in self.cfg.tokenizer_name.lower():
                    use_fast = False
                huggingface_token = os.environ.get("HF_TOKEN", None)
                add_bos_token = self.cfg.original_architecture not in [
                    "OlmoForCausalLM",
                    "OlmoeForCausalLM",
                ]
                self.set_tokenizer(
                    AutoTokenizer.from_pretrained(
                        self.cfg.tokenizer_name,
                        trust_remote_code=self.cfg.trust_remote_code,
                        use_fast=use_fast,
                        token=huggingface_token,
                        add_bos_token=add_bos_token,
                    ),
                    default_padding_side=default_padding_side,
                )
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and
            # will pass in tokens directly. In this case, we don't need a tokenizer.
            assert self.cfg.d_vocab != -1, "Must provide a tokenizer if d_vocab is not provided"
            self.tokenizer = None
            if default_padding_side != "right":
                logging.warning(
                    "default_padding_side is explictly given but ignored because tokenizer is not set."
                )

        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.cfg)
            self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]

        if self.cfg.use_hook_tokens:
            self.hook_tokens = HookPoint()  # [batch, pos]

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )

        if self.cfg.normalization_type == "RMS":
            self.ln_final = RMSNorm(self.cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln_final = RMSNormPre(self.cfg)
        elif self.cfg.normalization_type == "LN":
            if self.cfg.final_rms:
                self.ln_final = RMSNorm(self.cfg)
            else:
                self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            if self.cfg.final_rms:
                self.ln_final = RMSNormPre(self.cfg)
            else:
                self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning("Invalid normalization_type passed in %s", self.cfg.normalization_type)
        self.unembed = Unembed(self.cfg)

        if self.cfg.init_weights:
            self.init_weights()

        if move_to_device:
            # We load the devices in a pipeline manner - the first device gets the embed and
            # pos_embed layers and the first n_layers // n_devices blocks, the second gets the next
            # n_layers // n_devices blocks ... the last gets the last n_layers // n_devices blocks,
            # the final normalization layer (if it exists) and the unembed layer
            self.move_model_modules_to_device()

        # Helper variable to store a small (10K-20K) dataset of training data. Empty by default, can
        # be loaded with load_sample_training_dataset
        self.dataset = None

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    def check_hooks_to_add(
        self,
        hook_point,
        hook_point_name,
        hook,
        dir="fwd",
        is_permanent=False,
        prepend=False,
    ) -> None:
        if hook_point_name.endswith("attn.hook_result"):
            assert (
                self.cfg.use_attn_result
            ), f"Cannot add hook {hook_point_name} if use_attn_result_hook is False"
        if hook_point_name.endswith(("hook_q_input", "hook_k_input", "hook_v_input")):
            assert (
                self.cfg.use_split_qkv_input
            ), f"Cannot add hook {hook_point_name} if use_split_qkv_input is False"
        if hook_point_name.endswith("mlp_in"):
            assert (
                self.cfg.use_hook_mlp_in
            ), f"Cannot add hook {hook_point_name} if use_hook_mlp_in is False"
        if hook_point_name.endswith("attn_in"):
            assert (
                self.cfg.use_attn_in
            ), f"Cannot add hook {hook_point_name} if use_attn_in is False"

    def input_to_embed(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_model"],  # residual
        Optional[Int[torch.Tensor, "batch pos"]],  # tokens
        Optional[Float[torch.Tensor, "batch pos d_model"]],  # shortformer_pos_embed
        Optional[torch.Tensor],  # attention_mask [batch pos]
    ]:
        """Convert input to first residual stream.

        Args:
            input (Union[str, List[str], Int[torch.Tensor, "batch pos"]]): The input to the model.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side ([Literal["left", "right"], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing
                multiple strings of different lengths.
            past_kv_cache (HookedTransformerKeyValueCache, optional): If passed, we're doing caching
                and attention_mask will be stored in the cache.
        """
        if isinstance(input, str) or isinstance(input, list):
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # This is only intended to support passing in a single string
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input
        if len(tokens.shape) == 1:
            # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
            tokens = tokens[None]
        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(devices.get_device_for_block_index(0, self.cfg))

        if (
            (self.tokenizer and self.tokenizer.padding_side == "left")
            or attention_mask is not None
            or past_kv_cache is not None
        ):
            # This means we need to have an explicit attention mask.
            if attention_mask is None:
                # If the padding side is left or we are using caching, we need to compute the attention
                # mask for the adjustment of absolute positional embeddings and attention masking so
                # that pad tokens are not attended.
                if prepend_bos is USE_DEFAULT_VALUE:
                    prepend_bos = self.cfg.default_prepend_bos
                attention_mask = utils.get_attention_mask(self.tokenizer, tokens, prepend_bos)

            assert attention_mask.shape == tokens.shape, (
                f"Attention mask shape {attention_mask.shape} does not match tokens shape "
                f"{tokens.shape}"
            )
            attention_mask = attention_mask.to(devices.get_device_for_block_index(0, self.cfg))
            if past_kv_cache is not None:
                # past_kv_cache is not None, so we're doing caching.
                # We need to extend the previous attention_mask.
                # Update the past_kv_cache with the new attention_mask (unless it's frozen)
                attention_mask = past_kv_cache.append_attention_mask(attention_mask)
        else:
            # We separate this case from for computational efficiency.
            attention_mask = None

        # If we're doing caching, then we reuse keys and values from previous runs, as that's the
        # only way that past activations will affect the final logits. The cache contains those so
        # we don't need to recompute them. This is useful for generating text. As we have absolute
        # positional encodings, to implement this we have a `pos_offset` variable, defaulting to
        # zero, which says to offset which positional encodings are used (cached keys and values
        # were calculated with their own positional encodings).
        if past_kv_cache is None:
            pos_offset = 0
        else:
            batch_size, ctx_length = tokens.shape
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            if self.cfg.n_key_value_heads is None:
                assert num_heads_in_cache == self.cfg.n_heads
            else:
                assert num_heads_in_cache == self.cfg.n_key_value_heads
            assert d_head_in_cache == self.cfg.d_head
            pos_offset = cache_ctx_length
        if self.cfg.use_hook_tokens:
            tokens = self.hook_tokens(tokens)
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        if self.cfg.positional_embedding_type == "standard":
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, attention_mask)
            )  # [batch, pos, d_model]
            residual = embed + pos_embed  # [batch, pos, d_model]
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "shortformer":
            # If we're using shortformer style attention, we don't add the positional embedding to
            # the residual stream. See HookedTransformerConfig for details
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, attention_mask)
            )  # [batch, pos, d_model]
            residual = embed
            shortformer_pos_embed = pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            # Rotary doesn't use positional embeddings, instead they're applied when dot producting
            # keys and queries. See HookedTransformerConfig for details
            residual = embed
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "alibi":
            # ALiBi does not add positional embeddings to word embeddings,instead it biases QK attention scores.
            residual = embed
            shortformer_pos_embed = None
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )
        return residual, tokens, shortformer_pos_embed, attention_mask

    @overload
    def forward(
        self,
        input,
        return_type: Literal["logits"],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Loss:
        ...

    @overload
    def forward(
        self,
        input,
        return_type: Literal["loss"],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Loss:
        ...

    @overload
    def forward(
        self,
        input,
        return_type: Literal["both"],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss]:
        ...

    @overload
    def forward(
        self,
        input,
        return_type: Literal[None],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> None:
        ...

    def forward(
        self,
        input: Union[
            str,
            List[str],
            Int[torch.Tensor, "batch pos"],
            Float[torch.Tensor, "batch pos d_model"],
        ],
        return_type: Optional[str] = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        """Forward Pass.

        Input is either a batch of tokens ([batch, pos]) or a text string, a string is automatically
        tokenized to a batch of a single element. The prepend_bos flag only applies when inputting a
        text string.

        Note that loss is the standard "predict the next token" cross-entropy loss for GPT-2 style
        language models - if you want a custom loss function, the recommended behaviour is returning
        the logits and then applying your custom loss function.

        Args:
            return_type Optional[str]: The type of output to return. Can be one of: None (return
                nothing, don't calculate logits), 'logits' (return logits), 'loss' (return
                cross-entropy loss), 'both' (return logits and loss).
            loss_per_token bool: Whether to return the (next token prediction) loss per token (True)
                or average (False). Average loss is a scalar (averaged over position *and* batch),
                per-token loss is a tensor ([batch, position-1]) - position-1 because we're
                predicting the next token, and there's no specified next token for the final token.
                Defaults to False.
            prepend_bos Optional[bool]: Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. (Even for models not explicitly trained with a prepended BOS token, heads
                often use the first position as a resting position and accordingly lose information
                from the first token, so this empirically seems to give better results.) Pass True
                or False to locally override the default.
            padding_side Optional[Literal["left", "right"]]: Overrides self.tokenizer.padding_side.
                Specifies which side to pad on when tokenizing multiple strings of different
                lengths.
            start_at_layer Optional[int]: If not None, start the forward pass at the specified
                layer. Requires input to be the residual stream before the specified layer with
                shape [batch, pos, d_model]. Inclusive - ie, start_at_layer = 0 skips the embedding
                then runs the rest of the model. Supports negative indexing. start_at_layer = -1
                only runs the final block and the unembedding. Defaults to None (run the full
                model).
            tokens: Optional[Int[torch.Tensor, "batch pos"]]: Tokenized input. Only use if
                start_at_layer is not None and return type is "loss" or "both".
            shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]]: Positional
                embedding for shortformer models. Only use if start_at_layer is not None and
                self.cfg.positional_embedding_type == "shortformer".
            attention_mask: Optional[torch.Tensor]: Override the attention mask used to ignore
                padded tokens. If start_at_layer is not None and (self.tokenizer.padding_side ==
                "left" or past_kv_cache is not None), this should be passed as the attention mask
                is not computed automatically. Defaults to None.
            stop_at_layer Optional[int]: If not None, stop the forward pass at the specified layer.
                Exclusive - ie, stop_at_layer = 0 will only run the embedding layer, stop_at_layer =
                1 will run the embedding layer and the first transformer block, etc. Supports
                negative indexing. Useful for analysis of intermediate layers, eg finding neuron
                activations in layer 3 of a 24 layer model. Defaults to None (run the full model).
                If not None, we return the last residual stream computed.
            past_kv_cache Optional[HookedTransformerKeyValueCache]: If not None, keys and values
                will be stored for every attention head (unless the cache is frozen). If there are
                keys and values already in the cache, these will be prepended to the keys and values
                for the new input, so that the new tokens can pay attention to previous tokens. This
                is useful for generating text, because we don't need to repeat computation for
                tokens that have already been through the model. Also caches attention_mask so
                previous tokens are masked correctly (unless frozen). Padding should be ignored in
                all cases, so it's okay to eg. pass in left padded tokens twice in a row.
                Warning: Don't accidentally prepend_bos to the second half of a prompt.
                Defaults to None (don't use caching).
        """

        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if start_at_layer is None:
                (
                    residual,
                    tokens,
                    shortformer_pos_embed,
                    attention_mask,
                ) = self.input_to_embed(
                    input,
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                    attention_mask=attention_mask,
                    past_kv_cache=past_kv_cache,
                )
            else:
                assert type(input) == torch.Tensor
                residual = input

            if start_at_layer is None:
                start_at_layer = 0
            # If we explicitly want to start or stop at a layer, we only iterate through the blocks
            # between those indices. Note that start_at_layer is inclusive and stop_at_layer is
            # exclusive.
            # Eg: start_at_layer==None + stop_at_layer==0 means to only run the embed.
            # Eg: start_at_layer==3 + stop_at_layer==-1 means to run from layer 3 until the end of the PENULTIMATE layer
            blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
            for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:  # type: ignore
                # Note that each block includes skip connections, so we don't need
                # residual + block(residual)
                # If we're using multiple GPUs, we need to send the residual and shortformer_pos_embed to the correct GPU
                residual = residual.to(devices.get_device_for_block_index(i, self.cfg))
                if shortformer_pos_embed is not None:
                    shortformer_pos_embed = shortformer_pos_embed.to(
                        devices.get_device_for_block_index(i, self.cfg)
                    )

                residual = block(
                    residual,
                    # Cache contains a list of HookedTransformerKeyValueCache objects, one for each
                    # block
                    past_kv_cache_entry=past_kv_cache[i] if past_kv_cache is not None else None,
                    shortformer_pos_embed=shortformer_pos_embed,
                    attention_mask=attention_mask,
                )  # [batch, pos, d_model]

            if stop_at_layer is not None:
                # When we stop at an early layer, we end here rather than doing further computation
                return residual

            if self.cfg.normalization_type is not None:
                residual = self.ln_final(residual)  # [batch, pos, d_model]
            if return_type is None:
                return None
            else:
                logits = self.unembed(residual)  # [batch, pos, d_vocab]
                if self.cfg.output_logits_soft_cap > 0.0:
                    logits = self.cfg.output_logits_soft_cap * F.tanh(
                        logits / self.cfg.output_logits_soft_cap
                    )
                if return_type == "logits":
                    return logits
                else:
                    assert (
                        tokens is not None
                    ), "tokens must be passed in if return_type is 'loss' or 'both'"
                    loss = self.loss_fn(logits, tokens, attention_mask, per_token=loss_per_token)
                    if return_type == "loss":
                        return loss
                    elif return_type == "both":
                        return Output(logits, loss)
                    else:
                        logging.warning(f"Invalid return_type passed in: {return_type}")
                        return None

    def loss_fn(
        self,
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        tokens: Int[torch.Tensor, "batch pos"],
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        per_token: bool = False,
    ):
        """Wrapper around `utils.lm_cross_entropy_loss`.

        Used in forward() with return_type=="loss" or "both".
        """
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        return utils.lm_cross_entropy_loss(logits, tokens, attention_mask, per_token)

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Output, ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Output, Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around `run_with_cache` in HookedRootModule.

        If return_cache_object is True, this will return an ActivationCache object, with a bunch of
        useful HookedTransformer specific methods, otherwise it will return a dictionary of
        activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def set_tokenizer(
        self,
        tokenizer,
        default_padding_side="right",
    ):
        """Set the tokenizer to use for this model.

        Args:
            tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer.
            default_padding_side (str): "right" or "left", which side to pad on.

        """
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"

        assert default_padding_side in [
            "right",
            "left",
        ], f"padding_side must be 'right' or 'left', got {default_padding_side}"

        # Use a tokenizer that is initialized with add_bos_token=True as the default tokenizer.
        # Such a tokenizer should be set as the default tokenizer because the tokenization of some
        # tokenizers like LlamaTokenizer are different when bos token is automatically/manually
        # prepended, and add_bos_token cannot be dynamically controlled after initialization
        # (https://github.com/huggingface/transformers/issues/25886).
        if self.cfg.original_architecture not in [
            "OlmoForCausalLM",
            "OlmoeForCausalLM",
        ]:
            tokenizer_with_bos = utils.get_tokenizer_with_bos(tokenizer)
        else:
            tokenizer_with_bos = tokenizer
        self.tokenizer = tokenizer_with_bos
        assert self.tokenizer is not None  # keep mypy happy
        self.tokenizer.padding_side = default_padding_side

        # Some tokenizers doesn't automatically prepend the BOS token even when they are initialized
        # with add_bos_token=True. Therefore, we need this information to dynamically control prepend_bos.
        self.cfg.tokenizer_prepends_bos = len(self.tokenizer.encode("")) > 0

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        # Infer vocab size from tokenizer
        if self.cfg.d_vocab == -1:
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Int[torch.Tensor, "batch pos"]:
        """Converts a string to a tensor of tokens.

        If prepend_bos is True, prepends the BOS token to the input - this is recommended when
        creating a sequence of tokens to be input to a model.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when
        inputting a prompt to the model as the first token is often treated weirdly, but should only
        be done at the START of the prompt. Make sure to turn it off if you're looking at the
        tokenization of part of the prompt! (Note: some models eg GPT-2 were not trained with a BOS
        token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether
        the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not
        careful!

        Args:
            input (Union[str, List[str]]): The input to tokenize.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing
                multiple strings of different lengths.
            move_to_device (bool): Whether to move the output tensor of tokens to the device the
                model lives on. Defaults to True truncate (bool): If the output tokens are too long,
                whether to truncate the output tokens to the model's max context window. Does nothing
                for shorter inputs. Defaults to True.
        """
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
            assert (
                self.cfg.tokenizer_prepends_bos is not None
            ), "Set the tokenizer for the model by calling set_tokenizer"

            if self.cfg.default_prepend_bos and not self.cfg.tokenizer_prepends_bos:
                # We want to prepend bos but the tokenizer doesn't automatically do it, so we add it manually
                input = utils.get_input_with_manually_prepended_bos(self.tokenizer, input)

            tokens = self.tokenizer(
                input,
                return_tensors="pt",
                padding=True,
                truncation=truncate,
                max_length=self.cfg.n_ctx if truncate else None,
            )["input_ids"]

            if not self.cfg.default_prepend_bos and self.cfg.tokenizer_prepends_bos:
                # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
                tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)

            if move_to_device:
                tokens = tokens.to(self.cfg.device)
            return tokens

    def to_string(
        self,
        tokens: Union[
            List[int],
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "batch pos"],
            Int[torch.Tensor, "pos"],
            np.ndarray,
            List[Int[torch.Tensor, "pos"]],
        ],
    ) -> Union[str, List[str]]:
        """Tokens to String(s).

        Converts a tensor of tokens to a string (if rank 1) or a list of strings (if rank 2).

        Accepts lists of tokens and numpy arrays as inputs too (and converts to tensors internally)
        """
        assert self.tokenizer is not None, "Cannot use to_string without a tokenizer"

        if not isinstance(tokens, torch.Tensor):
            # We allow lists to be input
            tokens = torch.tensor(tokens)

        # I'm not sure what exactly clean_up_tokenization_spaces does, but if
        # it's set, then tokenization is no longer invertible, and some tokens
        # with a bunch of whitespace get collapsed together
        if len(tokens.shape) == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passed in: {tokens.shape}")

    def to_str_tokens(
        self,
        input: Union[
            str,
            Int[torch.Tensor, "pos"],
            Int[torch.Tensor, "1 pos"],
            Int[np.ndarray, "pos"],
            Int[np.ndarray, "1 pos"],
            list,
        ],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
    ) -> Union[List[str], List[List[str]]]:
        """Map text, a list of text or tokens to a list of tokens as strings.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when
        inputting a prompt to the model as the first token is often treated weirdly, but should only
        be done at the START of the prompt. If prepend_bos=None is passed, it implies the usage of
        self.cfg.default_prepend_bos which is set to True unless specified otherwise. Therefore,
        make sure to locally turn it off by passing prepend_bos=False if you're looking at the
        tokenization of part of the prompt! (Note: some models eg GPT-2 were not trained with a BOS
        token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether
        the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not
        careful!

        Gotcha3: If passing a string that exceeds the model's context length (model.cfg.n_ctx), it
        will be truncated.

        Args:
            input (Union[str, list, torch.Tensor]): The input - either a string or a tensor of
                tokens. If tokens, should be a tensor of shape [pos] or [1, pos].
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.

        Returns:
            str_tokens: List of individual tokens as strings
        """
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert self.tokenizer is not None  # keep mypy happy
            tokens: Union[np.ndarray, torch.Tensor]
            if isinstance(input, list):
                return list(
                    map(
                        lambda tokens: self.to_str_tokens(tokens, prepend_bos, padding_side),
                        input,
                    )
                )  # type: ignore
            elif isinstance(input, str):
                tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)[
                    0
                ]
                # Gemma tokenizer expects a batch dimension
                if "gemma" in self.tokenizer.name_or_path and tokens.ndim == 1:
                    tokens = tokens.unsqueeze(1)
            elif isinstance(input, torch.Tensor):
                tokens = input
                tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
                if tokens.dim() == 0:
                    # Don't pass dimensionless tensor
                    tokens = tokens.unsqueeze(0)
                assert (
                    tokens.dim() == 1
                ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
            elif isinstance(input, np.ndarray):
                tokens = input
                tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
                if tokens.ndim == 0:
                    # Don't pass dimensionless tensor
                    tokens = np.expand_dims(tokens, axis=0)
                assert (
                    tokens.ndim == 1
                ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
            else:
                raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
            str_tokens = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
            return str_tokens

    def to_single_token(self, string):
        """Map a string that makes up a single token to the id for that token.

        Raises an error for strings that are not a single token! If uncertain use to_tokens.
        """

        # We use the to_tokens method, do not append a BOS token
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        # If token shape is non-empty, raise error
        assert not token.shape, f"Input string: {string} is not a single token!"
        return token.item()

    def to_single_str_token(self, int_token: int) -> str:
        # Gives the single token corresponding to an int in string form
        assert isinstance(int_token, int)
        token = self.to_str_tokens(torch.tensor([int_token]))
        assert len(token) == 1
        return cast(str, token[0])

    def get_token_position(
        self,
        single_token: Union[str, int],
        input: Union[str, Union[Float[torch.Tensor, "pos"], Float[torch.Tensor, "1 pos"]]],
        mode="first",
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
    ):
        """Get the position of a single_token in a string or sequence of tokens.

        Raises an error if the token is not present.

        Gotcha: If you're inputting a string, it'll automatically be tokenized. Be careful about the
        setting for prepend_bos! When a string is input to the model, a BOS (beginning of sequence)
        token is prepended by default when the string is tokenized because
        self.cfg.default_prepend_bos is set to True unless specified otherwise. But this should only
        be done at the START of the input, not when inputting part of the prompt. If you're getting
        weird off-by-one errors, check carefully for what the setting should be!

        Args:
            single_token (Union[str, int]): The token to search for. Can
                be a token index, or a string (but the string must correspond to a single token).
            input (Union[str, torch.Tensor]): The sequence to
                search in. Can be a string or a rank 1 tensor of tokens or a rank 2 tensor of tokens
                with a dummy batch dimension.
            mode (str, optional): If there are multiple matches, which match to return. Supports
                "first" or "last". Defaults to "first".
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
        """
        if isinstance(input, str):
            # If the input is a string, convert to tensor
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input

        if len(tokens.shape) == 2:
            # If the tokens have shape [1, seq_len], flatten to [seq_len]
            assert (
                tokens.shape[0] == 1
            ), f"If tokens are rank two, they must have shape [1, seq_len], not {tokens.shape}"
            tokens = tokens[0]

        if isinstance(single_token, str):
            # If the single token is a string, convert to an integer
            single_token = self.to_single_token(single_token)
        elif isinstance(single_token, torch.Tensor):
            single_token = single_token.item()

        indices = torch.arange(len(tokens), device=tokens.device)[tokens == single_token]
        assert len(indices) > 0, "The token does not occur in the prompt"
        if mode == "first":
            return indices[0].item()
        elif mode == "last":
            return indices[-1].item()
        else:
            raise ValueError(f"mode must be 'first' or 'last', not {mode}")

    def tokens_to_residual_directions(
        self,
        tokens: Union[
            str,
            int,
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "pos"],
            Int[torch.Tensor, "batch pos"],
        ],
    ) -> Union[
        Float[torch.Tensor, "d_model"],
        Float[torch.Tensor, "pos d_model"],
        Float[torch.Tensor, "batch pos d_model"],
    ]:
        """Map tokens to a tensor with the unembedding vector for those tokens.

        I.e. the vector in the residual stream that we dot with to the get the logit for that token.

        WARNING: If you use this without folding in LayerNorm, the results will be misleading and
        may be incorrect, as the LN weights change the unembed map. This is done automatically with
        the fold_ln flag on from_pretrained

        WARNING 2: LayerNorm scaling will scale up or down the effective direction in the residual
        stream for each output token on any given input token position.
        ActivationCache.apply_ln_to_stack will apply the appropriate scaling to these directions.

        Args:
            tokens (Union[str, int, torch.Tensor]): The token(s). If a single token, can be a single
                element tensor, an integer, or string. If string, will be mapped to a single token
                using to_single_token, and an error raised if it's multiple tokens. The method also
                works for a batch of input tokens.

        Returns:
            residual_direction torch.Tensor: The unembedding vector for the token(s), a stack of
                [d_model] tensor.
        """
        if isinstance(tokens, torch.Tensor) and tokens.numel() > 1:
            # If the tokens are a tensor, and have more than one element, assume they are a batch of
            # tokens.
            residual_directions = self.W_U[:, tokens]
            residual_directions = einops.rearrange(
                residual_directions, "d_model ... -> ... d_model"
            )
            return residual_directions
        else:
            # Otherwise there is a single token
            if isinstance(tokens, str):
                token = self.to_single_token(tokens)
            elif isinstance(tokens, int):
                token = tokens
            elif isinstance(tokens, torch.Tensor) and tokens.numel() == 1:
                token = tokens.item()
            else:
                raise ValueError(f"Invalid token type: {type(tokens)}")
            residual_direction = self.W_U[:, token]
            return residual_direction

    def to(  # type: ignore
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):
        """Wrapper around cuda that also changes `self.cfg.device`."""
        return self.to("cuda")

    def cpu(self):
        """Wrapper around cuda that also changes `self.cfg.device`."""
        return self.to("cpu")

    def mps(self):
        """Wrapper around mps that also changes `self.cfg.device`."""
        return self.to("mps")

    def move_model_modules_to_device(self):
        self.embed.to(devices.get_device_for_block_index(0, self.cfg))
        self.hook_embed.to(devices.get_device_for_block_index(0, self.cfg))
        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed.to(devices.get_device_for_block_index(0, self.cfg))
            self.hook_pos_embed.to(devices.get_device_for_block_index(0, self.cfg))

        if hasattr(self, "ln_final"):
            self.ln_final.to(devices.get_device_for_block_index(self.cfg.n_layers - 1, self.cfg))
        self.unembed.to(devices.get_device_for_block_index(self.cfg.n_layers - 1, self.cfg))
        for i, block in enumerate(self.blocks):
            block.to(devices.get_device_for_block_index(i, self.cfg))

    @classmethod
    def from_pretrained(
        cls: Type[T],
        model_name: str,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        refactor_factored_attn_matrices: bool = False,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model: Optional[AutoModelForCausalLM] = None,
        device: Optional[Union[str, torch.device]] = None,
        n_devices: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        move_to_device: bool = True,
        fold_value_biases: bool = True,
        default_prepend_bos: Optional[bool] = None,
        default_padding_side: Literal["left", "right"] = "right",
        dtype="float32",
        first_n_layers: Optional[int] = None,
        **from_pretrained_kwargs,
    ) -> T:
        """Load in a Pretrained Model.

        Load in pretrained model weights to the HookedTransformer format and optionally to do some
        processing to make the model easier to interpret. Currently supports loading from most
        autoregressive HuggingFace models (``gpt2``, ``neo``, ``gptj``, ``opt``...) and from a range
        of toy models and SoLU models trained by Neel Nanda. The full list is available in the docs
        under :doc:`model properties</generated/model_properties_table>`. Also supports loading from
        a checkpoint for checkpointed models (currently, models trained by NeelNanda and the
        stanford-crfm models (using parameters ``checkpoint_index`` and ``checkpoint_value``).

        See :meth:`load_and_process_state_dict` for details on the processing (folding layer norm,
        centering the unembedding and centering the writing weights).

        Example:

        >>> from transformer_lens import HookedTransformer
        >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
        Loaded pretrained model tiny-stories-1M into HookedTransformer

        Args:
            model_name: The model name - must be an element of
                :const:`transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES` or an alias
                of one. The full list of available models can be found in the docs under :doc:`model
                properties</generated/model_properties_table>`.
            fold_ln: Whether to fold in the LayerNorm weights to the
                subsequent linear layer. This does not change the computation.

                `LayerNorm
                <https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1>`_
                is a common regularization technique used in transformers. Unlike BatchNorm, it
                cannot be turned off at inference time, as it significantly alters the mathematical
                function implemented by the transformer.

                When `fold_ln` is set to True, LayerNorm (with weights :math:`w_{ln}` and
                :math:`b_{ln}`) followed by a linear layer (:math:`W + b`) is optimized to
                LayerNormPre (just centering & normalizing) followed by a new linear layer with
                :math:`W_{eff} = w[:, \text{None}] * W` (element-wise multiplication) and
                :math:`b_{eff} = b + b_{ln} @ W`. This transformation is computationally equivalent
                and simplifies the model's interpretability. It essentially merges LayerNorm weights
                into the subsequent linear layer's weights, which is handled by HookedTransformer
                when loading pre-trained weights. Set `fold_ln` to False when loading a state dict
                if you wish to turn this off.

                Mathematically, LayerNorm is defined as follows:

                .. math::
                    x_1 &= x_0 - \\text{mean}(x_0)

                    x_2 &= \\frac{x_1}{\\sqrt{\\text{mean}(x_1^2)}}

                    x_3 &= x_2 \\cdot w

                    x_4 &= x_3 + b

                For further details, refer to `this document
                <https://transformer-circuits.pub/2021/framework/index.html#:~:text=Handling%20Layer%20Normalization>`_.
            center_writing_weights: Whether to center weights
                writing to the residual stream (ie set mean to be zero). Due to LayerNorm this
                doesn't change the computation.

                A related idea to folding layernorm (``fold_ln``) - *every* component reading an
                input from the residual stream is preceded by a LayerNorm, which means that the mean
                of a residual stream vector (ie the component in the direction of all ones) never
                matters. This means we can remove the all ones component of weights and biases whose
                output *writes* to the residual stream. Mathematically, ``W_writing -=
                W_writing.mean(dim=1, keepdim=True)``.
            center_unembed: Whether to center W_U (ie set mean
                to be zero). Softmax is translation invariant so this doesn't affect log probs or
                loss, but does change logits.

                The logits are fed into a softmax. Softmax is translation invariant (eg, adding 1 to
                every logit doesn't change the output), so we can simplify things by setting the
                mean of the logits to be zero. This is equivalent to setting the mean of every
                output vector of ``W_U`` to zero. In code, ``W_U -= W_U.mean(dim=-1,
                keepdim=True)``.
            refactor_factored_attn_matrices: Whether to convert the factored
                matrices (W_Q & W_K, and W_O & W_V) to be "even". Defaults to False
            checkpoint_index: If loading from a checkpoint, the index of
                the checkpoint to load.
            checkpoint_value: If loading from a checkpoint, the value of
                the checkpoint to load, ie the step or token number (each model has checkpoints
                labelled with exactly one of these). E.g. ``1000`` for a checkpoint taken at step
                1000 or after 1000 tokens. If `checkpoint_index` is also specified, this will be
                ignored.
            hf_model: If you have already loaded in the
                HuggingFace model, you can pass it in here rather than needing to recreate the
                object. Defaults to None.
            device: The device to load the model onto. By
                default will load to CUDA if available, else CPU.
            n_devices: The number of devices to split the model
                across. Defaults to 1. If greater than 1, `device` must be cuda.
            tokenizer: The tokenizer to use for the model. If not
                provided, it is inferred from cfg.tokenizer_name or initialized to None. If None,
                then the model cannot be passed strings, and d_vocab must be explicitly set.
            move_to_device: Whether to move the model to the device specified in
                cfg. device. Must be true if `n_devices` in the config is greater than 1, since the
                model's layers will be split across multiple devices.
            fold_value_biases: Each attention head has a value bias. Values are averaged to create
                mixed values (``z``), weighted by the attention pattern, but as the bias is
                constant, its contribution to ``z`` is exactly the same. The output of a head is ``z
                @ W_O``, and so the value bias just linearly adds to the output of the head. This
                means that the value bias of a head has nothing to do with the head, and is just a
                constant added to the attention layer outputs. We can take the sum across these and
                b_O to get an "effective bias" for the layer. In code, we set ``b_V=0``. and ``b_O =
                (b_V @ W_O).sum(dim=0) + b_O``.

                The technical derivation of this is as follows. ``v = residual @ W_V[h] +
                broadcast_b_V[h]`` for each head ``h`` (where ``b_V`` is broadcast up from shape
                ``d_head`` to shape ``[position, d_head]``). And ``z = pattern[h] @ v = pattern[h] @
                residual @ W_V[h] + pattern[h] @ broadcast_b_V[h]``. Because ``pattern[h]`` is
                ``[destination_position, source_position]`` and ``broadcast_b_V`` is constant along
                the ``(source_)position`` dimension, we're basically just multiplying it by the sum
                of the pattern across the ``source_position`` dimension, which is just ``1``. So it
                remains exactly the same, and so is just broadcast across the destination positions.
            default_prepend_bos: Default behavior of whether to prepend the BOS
                token when the methods of HookedTransformer process input text to tokenize (only
                when input is a string).
                Resolution order for default_prepend_bos:
                1. If user passes value explicitly, use that value
                2. Model-specific default from cfg_dict if it exists (e.g. for bloom models it's False)
                3. Global default (True)

                Even for models not explicitly trained with the BOS token, heads often use the first position as a resting position
                and accordingly lose information from the first token, so this empirically seems to give better
                results. Note that you can also locally override the default behavior by passing in
                prepend_bos=True/False when you call a method that processes the input string.
            from_pretrained_kwargs: Any other optional argument passed to
                HuggingFace's from_pretrained (e.g. "cache_dir" or "torch_dtype"). Also passed to
                other HuggingFace functions when compatible. For some models or arguments it doesn't
                work, especially for models that are not internally loaded with HuggingFace's
                from_pretrained (e.g. SoLU models).
            dtype: What data type to load the model in (also sets the dtype of
                the HuggingFace model). Set to bfloat16 or float16 if you get out of memory errors when loading
                the model.
            default_padding_side: Which side to pad on when tokenizing. Defaults to
                "right".
            first_n_layers: If specified, only load the first n layers of the model.
        """
        if model_name.lower().startswith("t5"):
            raise RuntimeError(
                "Execution stopped: Please use HookedEncoderDecoder to load T5 models instead of HookedTransformer."
            )

        assert not (
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if hf_model is not None:
            hf_cfg = hf_model.config.to_dict()
            qc = hf_cfg.get("quantization_config", {})
            load_in_4bit = qc.get("load_in_4bit", False)
            load_in_8bit = qc.get("load_in_8bit", False)
            quant_method = qc.get("quant_method", "")
            assert not load_in_8bit, "8-bit quantization is not supported"
            assert not (
                load_in_4bit and (version.parse(torch.__version__) < version.parse("2.1.1"))
            ), "Quantization is only supported for torch versions >= 2.1.1"
            assert not (
                load_in_4bit and ("llama" not in model_name.lower())
            ), "Quantization is only supported for Llama models"
            if load_in_4bit:
                assert (
                    qc.get("quant_method", "") == "bitsandbytes"
                ), "Only bitsandbytes quantization is supported"
        else:
            hf_cfg = {}

        if isinstance(dtype, str):
            # Convert from string to a torch dtype
            dtype = DTYPE_FROM_STRING[dtype]
        if "torch_dtype" in from_pretrained_kwargs:
            # For backwards compatibility with the previous way to do low precision loading
            # This should maybe check the user did not explicitly set dtype *and* torch_dtype
            dtype = from_pretrained_kwargs["torch_dtype"]

        if (
            (from_pretrained_kwargs.get("torch_dtype", None) == torch.float16)
            or dtype == torch.float16
        ) and device in ["cpu", None]:
            logging.warning("float16 models may not work on CPU. Consider using a GPU or bfloat16.")

        # Get the model name used in HuggingFace, rather than the alias.
        official_model_name = loading.get_official_model_name(model_name)

        # Load the config into an HookedTransformerConfig object. If loading from a
        # checkpoint, the config object will contain the information about the
        # checkpoint
        cfg = loading.get_pretrained_model_config(
            official_model_name,
            hf_cfg=hf_cfg,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=fold_ln,
            device=device,
            n_devices=n_devices,
            default_prepend_bos=default_prepend_bos,
            dtype=dtype,
            first_n_layers=first_n_layers,
            **from_pretrained_kwargs,
        )

        if cfg.positional_embedding_type == "shortformer":
            if fold_ln:
                logging.warning(
                    "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_"
                    "ln=False instead."
                )
                fold_ln = False
            if center_unembed:
                logging.warning(
                    "You tried to specify center_unembed=True for a shortformer model, but this can't be done! "
                    "Setting center_unembed=False instead."
                )
                center_unembed = False
            if center_writing_weights:
                logging.warning(
                    "You tried to specify center_writing_weights=True for a shortformer model, but this can't be done! "
                    "Setting center_writing_weights=False instead."
                )
                center_writing_weights = False
        if center_unembed and cfg.output_logits_soft_cap > 0.0:
            logging.warning(
                "You tried to specify center_unembed=True for a model using logit softcap, but this can't be done! Softcapping is not invariant upon adding a constant "
                "Setting center_unembed=False instead."
            )
            center_unembed = False

        # Get the state dict of the model (ie a mapping of parameter names to tensors), processed to
        # match the HookedTransformer parameter names.
        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )

        # Create the HookedTransformer object
        model = cls(
            cfg,
            tokenizer,
            move_to_device=False,
            default_padding_side=default_padding_side,
        )

        model.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )

        if move_to_device:
            model.move_model_modules_to_device()

        print(f"Loaded pretrained model {model_name} into HookedTransformer")

        return model

    @classmethod
    def from_pretrained_no_processing(
        cls,
        model_name: str,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        refactor_factored_attn_matrices=False,
        fold_value_biases=False,
        dtype=torch.float32,
        default_prepend_bos=None,
        default_padding_side="right",
        **from_pretrained_kwargs,
    ):
        """Wrapper for from_pretrained.

        Wrapper for from_pretrained with all boolean flags related to simplifying the model set to
        False. Refer to from_pretrained for details.
        """
        return cls.from_pretrained(
            model_name,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            dtype=dtype,
            default_prepend_bos=default_prepend_bos,
            default_padding_side=default_padding_side,
            **from_pretrained_kwargs,
        )

    def init_weights(self):
        """Initialize weights.

        LayerNorm weights are already initialized to 1.0, and all biases are initialized to 0.0
        (including LayerNorm), so this just initializes weight matrices.

        Weight matrices are set to empty by default (to save space + compute, since they're the bulk
        of the parameters), so it is important to call this if you are not loading in pretrained
        weights! Note that this function assumes that weight names being with `W_`.

        Set seed here to ensure determinism.

        This does NOT follow the PyTorch scheme, which as far as I can tell is super out of date but
        no one has gotten round to updating it? https://github.com/pytorch/pytorch/issues/18182

        The default PyTorch scheme is the following: all linear layers use uniform(-1/sqrt(fan_in),
        1/sqrt(fan_in)) for weights, and uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for biases. For
        biases, fan_in is computed using the fan_in for the weight matrix of the linear layer. Note
        tha it *does not actually* use Kaiming initialization, despite the fact that it calls the
        function.

        However, for Transformer blocks, it instead initializes biases to zero and weights using Xavier uniform, that
        is: uniform(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))) for weights.

        PyTorch Transformers are especially bad - TransformerEncoder initializes all layers to the
        exact same weights?! https://github.com/pytorch/pytorch/issues/72253.

        The best paper I've found on transformer initialization is the muP paper, but haven't
        integrated those ideas yet: https://arxiv.org/abs/2203.03466

        We split off the initialization into separate functions because muP initialization handles
        different parts of the model differently.
        """

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        if self.cfg.init_mode == "gpt2":
            self._init_weights_gpt2()
        elif self.cfg.init_mode == "xavier_uniform":
            self._init_weights_xavier(dist_type="uniform")
        elif self.cfg.init_mode == "xavier_normal":
            self._init_weights_xavier(dist_type="normal")
        elif self.cfg.init_mode == "kaiming_uniform":
            self._init_weights_kaiming(dist_type="uniform")
        elif self.cfg.init_mode == "kaiming_normal":
            self._init_weights_kaiming(dist_type="normal")
        elif self.cfg.init_mode == "muP":
            self._init_weights_muP(dist_type="normal")  # muP uses normal initialization

    def _init_weights_gpt2(self):
        """Initialize weights with GPT-2 initialization. Biases are initialized to 0.0 and weights
        are initialized to N(0, 0.64/d_model) if initializer_range is not set, otherwise std is initializer_range.
        """
        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=self.cfg.initializer_range)

    def _init_weights_xavier(self, dist_type="normal"):
        """
        Initialize weights with Xavier initialization -- that is, scale the weights by sqrt(6 /
        (fan_in + fan_out)) for a [-1, 1] uniform distribution, or sqrt(2 / (fan_in + fan_out)) for a
        standard normal.

        Note that since TransformerLens implements the matrices in the opposite orientation to what
        torch does (e.g. it's d_in x d_out, not d_out x d_in as in torch), we need to calculate it
        ourselves.
        """
        gain = self.cfg.initializer_range
        for name, param in self.named_parameters():
            if "W_" in name:
                if dist_type == "uniform":
                    init_xavier_uniform_(param, gain=gain)
                elif dist_type == "normal":
                    init_xavier_normal_(param, gain=gain)

    def _init_weights_kaiming(self, dist_type="uniform"):
        """
        Initialize weights with Kaiming initialization -- that is, scale the weights by
        c / sqrt(fan_in), where c = sqrt(2) if the params were immediately preceded by a relu and 1 for
        everything else.

        Note that the numbers are actually incorrect here when you're using a nonlinearity other
        than relu, e.g. the correct c for SiLu is ~1.74, for tanh it's 5/3 ~= 1.67, and for GeLU it's ~1.57.
        But this is unlikely to matter in practice.

        I'm just using fan_mode = "fan_in" for now, but it should be trivial to add fan_out.

        Again, we have to implement it ourselves because of the orientation of the matrices.
        """
        gain = self.cfg.initializer_range
        for name, param in self.named_parameters():
            if "W_" in name:
                if dist_type == "uniform":
                    init_kaiming_uniform_(param, gain=gain, nonlinearity="relu", mode="fan_in")
                elif dist_type == "normal":
                    init_kaiming_normal_(param, gain=gain, nonlinearity="relu", mode="fan_in")

    def _init_weights_muP(self, dist_type="uniform"):
        """
        Initialize weights with muParameterization. This involves scaling output weights by a factor
        of 1/fan_in, input weights and biases by 1, everything else by a factor of 1/sqrt(fan_in).

        Also, you need to use muAdamW, which rescales the learning rate for output weights and
        hidden weights by a factor of 1/fan_in.

        All biases are still assumed to be initialized to 0.0, so we only need to change the
        weights.
        """
        for name, param in self.named_parameters():
            if "W_" in name:
                fan_in, _ = utils.calc_fan_in_and_fan_out(param)
                if "embed" in name:
                    scale = float(1)
                elif "unembed" in name:
                    scale = 1 / fan_in
                else:
                    scale = 1 / fan_in**0.5

                if dist_type == "uniform":
                    scale *= 3**0.5
                    nn.init.uniform_(param, -scale, scale)
                elif dist_type == "normal":
                    nn.init.normal_(param, std=scale)

    def load_and_process_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ):
        """Load & Process State Dict.

        Load a state dict into the model, and to apply processing to simplify it. The state dict is
        assumed to be in the HookedTransformer format.

        See the relevant method (same name as the flag) for more details on the folding, centering
        and processing flags.

        Args:
            state_dict (dict): The state dict of the model, in HookedTransformer format. fold_ln
            fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the
                subsequent linear layer. This does not change the computation. Defaults to True.
            center_writing_weights (bool, optional): Whether to center weights writing to the
                residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the
                computation. Defaults to True.
            center_unembed (bool, optional): Whether to center W_U (ie set mean to be zero).
                Softmax is translation invariant so this doesn't affect log probs or loss, but does
                change logits. Defaults to True.
            fold_value_biases (bool, optional): Whether to fold the value biases into the output
                bias. Because attention patterns add up to 1, the value biases always have a
                constant effect on a layer's output, and it doesn't matter which head a bias is
                associated with. We can factor this all into a single output bias to the layer, and
                make it easier to interpret the head's output.
            refactor_factored_attn_matrices (bool, optional): Whether to convert the factored
                matrices (W_Q & W_K, and W_O & W_V) to be "even". Defaults to False.
            model_name (str, optional): checks the model name for special cases of state dict
                loading. Only used for Redwood 2L model currently.
        """
        if self.cfg.dtype not in [torch.float32, torch.float64] and fold_ln:
            logging.warning(
                "With reduced precision, it is advised to use `from_pretrained_no_processing` instead of `from_pretrained`."
            )

        if (
            self.cfg.dtype not in [torch.float32, torch.float64]
            and self.cfg.num_experts
            and self.cfg.num_experts > 1
        ):
            logging.warning(
                "When running MoE models, it is advised to use a higher precision data type. See docs for more info."
            )

        state_dict = self.fill_missing_keys(state_dict)
        if fold_ln:
            if self.cfg.num_experts and self.cfg.num_experts > 1:
                logging.warning(
                    "You are using MoE, so the layer norm weights can't be folded! Skipping"
                )
            elif self.cfg.normalization_type in ["LN", "LNPre"]:
                state_dict = self.fold_layer_norm(state_dict)
            elif self.cfg.normalization_type in ["RMS", "RMSPre"]:
                state_dict = self.fold_layer_norm(
                    state_dict, fold_biases=False, center_weights=False
                )
            else:
                logging.warning(
                    "You are not using LayerNorm or RMSNorm, so the layer norm weights can't be folded! Skipping"
                )

        if center_writing_weights:
            if self.cfg.normalization_type not in ["LN", "LNPre"]:
                logging.warning(
                    "You are not using LayerNorm, so the writing weights can't be centered! Skipping"
                )
            elif self.cfg.final_rms:
                logging.warning(
                    "This model is using final RMS normalization, so the writing weights can't be centered! Skipping"
                )
            else:
                state_dict = self.center_writing_weights(state_dict)

        if center_unembed:
            state_dict = self.center_unembed(state_dict)
        if fold_value_biases:
            state_dict = self.fold_value_biases(state_dict)
        if refactor_factored_attn_matrices:
            state_dict = self.refactor_factored_attn_matrices(state_dict)

        if self.cfg.load_in_4bit:
            # with quantization, parameters should be assigned
            # so that quantization settings are not lost
            self.load_state_dict(state_dict, assign=True, strict=False)
        else:
            state_dict_keys = list(state_dict.keys())
            for key in state_dict_keys:
                self.load_state_dict({key: state_dict[key]}, strict=False)
                del state_dict[key]

    def fill_missing_keys(self, state_dict):
        return loading.fill_missing_keys(self, state_dict)

    def fold_layer_norm(
        self, state_dict: Dict[str, torch.Tensor], fold_biases=True, center_weights=True
    ):
        """Fold Layer Norm. Can also be used to fold RMS Norm, when fold_biases and center_weights are set to False.

        Takes in a state dict from a pretrained model, formatted to be consistent with
        HookedTransformer but with LayerNorm weights and biases. Folds these into the neighbouring
        weights. See further_comments.md for more details.

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model.
            fold_biases (bool): Enables folding of LN biases. Should be disabled when RMS Norm is used.
            center_weights (bool): Enables the centering of weights after folding in LN. Should be disabled when RMS Norm is used.
        """

        # Models that use Grouped Query Attention (Only Mistral at the time of writing) prefix their K/V weights and
        # biases with an underscore in order to distinguish them, but folding the LN into them still works the same,
        # so we just add the underscore if GQA is used (i.e. if `cfg.n_key_value_heads is specified`).
        gqa = "" if self.cfg.n_key_value_heads is None else "_"

        for l in range(self.cfg.n_layers):
            # Fold ln1 into attention - it's important to fold biases first, since biases depend on
            # weights but not vice versa The various indexing is just to broadcast ln.b and ln.w
            # along every axis other than d_model. Each weight matrix right multiplies. To fold in
            # the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need
            # to sum along axis -2, which is the residual stream space axis.
            if fold_biases:
                state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[f"blocks.{l}.attn.b_Q"] + (
                    state_dict[f"blocks.{l}.attn.W_Q"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(-2)
                state_dict[f"blocks.{l}.attn.{gqa}b_K"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_K"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                state_dict[f"blocks.{l}.attn.{gqa}b_V"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_V"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(
                    -2
                )
                del state_dict[f"blocks.{l}.ln1.b"]

            state_dict[f"blocks.{l}.attn.W_Q"] = (
                state_dict[f"blocks.{l}.attn.W_Q"] * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_K"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_V"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            del state_dict[f"blocks.{l}.ln1.w"]

            # Finally, we center the weights reading from the residual stream. The output of the
            # first part of the LayerNorm is mean 0 and standard deviation 1, so the mean of any
            # input vector of the matrix doesn't matter and can be set to zero. Equivalently, the
            # output of LayerNormPre is orthogonal to the vector of all 1s (because dotting with
            # that gets the sum), so we can remove the component of the matrix parallel to this.
            if center_weights:
                state_dict[f"blocks.{l}.attn.W_Q"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_K"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_V"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )

            # Fold ln2 into MLP
            if not self.cfg.attn_only:
                if fold_biases:
                    state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[f"blocks.{l}.mlp.b_in"] + (
                        state_dict[f"blocks.{l}.mlp.W_in"]
                        * state_dict[f"blocks.{l}.ln2.b"][:, None]
                    ).sum(-2)
                    del state_dict[f"blocks.{l}.ln2.b"]

                state_dict[f"blocks.{l}.mlp.W_in"] = (
                    state_dict[f"blocks.{l}.mlp.W_in"] * state_dict[f"blocks.{l}.ln2.w"][:, None]
                )

                if self.cfg.gated_mlp:
                    state_dict[f"blocks.{l}.mlp.W_gate"] = (
                        state_dict[f"blocks.{l}.mlp.W_gate"]
                        * state_dict[f"blocks.{l}.ln2.w"][:, None]
                    )

                del state_dict[f"blocks.{l}.ln2.w"]

                if center_weights:
                    # Center the weights that read in from the LayerNormPre
                    state_dict[f"blocks.{l}.mlp.W_in"] -= einops.reduce(
                        state_dict[f"blocks.{l}.mlp.W_in"],
                        "d_model d_mlp -> 1 d_mlp",
                        "mean",
                    )

                if self.cfg.act_fn is not None and self.cfg.act_fn.startswith("solu"):
                    # Fold ln3 into activation
                    if fold_biases:
                        state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[
                            f"blocks.{l}.mlp.b_out"
                        ] + (
                            state_dict[f"blocks.{l}.mlp.W_out"]
                            * state_dict[f"blocks.{l}.mlp.ln.b"][:, None]
                        ).sum(
                            -2
                        )

                        del state_dict[f"blocks.{l}.mlp.ln.b"]

                    state_dict[f"blocks.{l}.mlp.W_out"] = (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.w"][:, None]
                    )

                    if center_weights:
                        # Center the weights that read in from the LayerNormPre
                        state_dict[f"blocks.{l}.mlp.W_out"] -= einops.reduce(
                            state_dict[f"blocks.{l}.mlp.W_out"],
                            "d_mlp d_model -> 1 d_model",
                            "mean",
                        )

                    del state_dict[f"blocks.{l}.mlp.ln.w"]

        # Fold ln_final into Unembed
        if not self.cfg.final_rms and fold_biases:
            # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm
            # pre unembed.
            state_dict["unembed.b_U"] = state_dict["unembed.b_U"] + (
                state_dict["unembed.W_U"] * state_dict["ln_final.b"][:, None]
            ).sum(dim=-2)
            del state_dict["ln_final.b"]

        state_dict["unembed.W_U"] = state_dict["unembed.W_U"] * state_dict["ln_final.w"][:, None]
        del state_dict["ln_final.w"]

        if center_weights:
            # Center the weights that read in from the LayerNormPre
            state_dict["unembed.W_U"] -= einops.reduce(
                state_dict["unembed.W_U"], "d_model d_vocab -> 1 d_vocab", "mean"
            )

        return state_dict

    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Center Writing Weights.

        Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and
        W_out. This is done by subtracting the mean of the weights from the weights themselves. This
        is done in-place. See fold_layer_norm for more details.
        """
        state_dict["embed.W_E"] = state_dict["embed.W_E"] - state_dict["embed.W_E"].mean(
            -1, keepdim=True
        )
        if self.cfg.positional_embedding_type != "rotary":
            state_dict["pos_embed.W_pos"] = state_dict["pos_embed.W_pos"] - state_dict[
                "pos_embed.W_pos"
            ].mean(-1, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f"blocks.{l}.attn.W_O"] = state_dict[f"blocks.{l}.attn.W_O"] - state_dict[
                f"blocks.{l}.attn.W_O"
            ].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[f"blocks.{l}.attn.b_O"] = (
                state_dict[f"blocks.{l}.attn.b_O"] - state_dict[f"blocks.{l}.attn.b_O"].mean()
            )  # b_O is [d_model]
            if not self.cfg.attn_only:
                state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[
                    f"blocks.{l}.mlp.W_out"
                ] - state_dict[f"blocks.{l}.mlp.W_out"].mean(-1, keepdim=True)
                state_dict[f"blocks.{l}.mlp.b_out"] = (
                    state_dict[f"blocks.{l}.mlp.b_out"] - state_dict[f"blocks.{l}.mlp.b_out"].mean()
                )
        return state_dict

    def center_unembed(self, state_dict: Dict[str, torch.Tensor]):
        """Center the unembedding weights W_U.

        This is done by subtracting the mean of the weights from the weights themselves. This is
        done in-place. As softmax is translation invariant, this changes the logits but not the log
        probs, and makes the model logits (slightly) more interpretable - when trying to understand
        how components contribute to the logits, we'll be less misled by components that just add
        something to every logit.
        """
        state_dict["unembed.W_U"] = state_dict["unembed.W_U"] - state_dict["unembed.W_U"].mean(
            -1, keepdim=True
        )
        state_dict["unembed.b_U"] = state_dict["unembed.b_U"] - state_dict["unembed.b_U"].mean()
        return state_dict

    def fold_value_biases(self, state_dict: Dict[str, torch.Tensor]):
        """Fold the value biases into the output bias.

        Because attention patterns add up to 1, the value biases always have a constant effect on a
        head's output. Further, as the outputs of each head in a layer add together, each head's
        value bias has a constant effect on the *layer's* output, which can make it harder to
        interpret the effect of any given head, and it doesn't matter which head a bias is
        associated with. We can factor this all into a single output bias to the layer, and make it
        easier to interpret the head's output. Formally, we take b_O_new = b_O_original +
        sum_head(b_V_head @ W_O_head).
        """
        for layer in range(self.cfg.n_layers):
            # shape [head_index, d_head]
            if self.cfg.n_key_value_heads is None:
                b_V = state_dict[f"blocks.{layer}.attn.b_V"]
            else:
                b_V = state_dict[f"blocks.{layer}.attn._b_V"]
                b_V = torch.repeat_interleave(
                    b_V, dim=0, repeats=self.cfg.n_heads // self.cfg.n_key_value_heads
                )
            # [head_index, d_head, d_model]
            W_O = state_dict[f"blocks.{layer}.attn.W_O"]
            # [d_model]
            b_O_original = state_dict[f"blocks.{layer}.attn.b_O"]
            folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])

            state_dict[f"blocks.{layer}.attn.b_O"] = folded_b_O
            if self.cfg.n_key_value_heads is None:
                state_dict[f"blocks.{layer}.attn.b_V"] = torch.zeros_like(b_V)
            else:
                state_dict[f"blocks.{layer}.attn._b_V"] = torch.zeros_like(
                    state_dict[f"blocks.{layer}.attn._b_V"]
                )
        return state_dict

    def refactor_factored_attn_matrices(self, state_dict: Dict[str, torch.Tensor]):
        """Experimental method for managing queries, keys and values.

        As argued in [A Mathematical Framework for Transformer
        Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and
        values are somewhat arbitrary intermediate terms when computing with the low rank factored
        matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing
        determining head behaviour. But there are many ways to find a low rank factorization to a
        given matrix, and hopefully some of these are more interpretable than others! This method is
        one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a
        rotation and W_Q and W_K having the nth column in each having the same norm. The formula is
        $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not
        R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank
        factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and
        $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now
        $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the
        result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$,
        which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the
        matrices as having the same norm, since there's not an obvious asymmetry between the keys
        and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V)
        @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that
        b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along
        the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q + b_Q) * (y @ W_K +
        b_K), which is fairly messy. To deal with the biases, we concatenate them to W_Q and W_K to
        simulate a d_model+1 dimensional input (whose final coordinate is always 1), do the SVD
        factorization on this effective matrix, then separate out into final weights and biases.
        """

        assert (
            self.cfg.positional_embedding_type != "rotary"
        ), "You can't refactor the QK circuit when using rotary embeddings (as the QK matrix depends on the position of the query and key)"

        for l in range(self.cfg.n_layers):
            # W_QK = W_Q @ W_K.T
            # Concatenate biases to make a d_model+1 input dimension
            W_Q_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    state_dict[f"blocks.{l}.attn.b_Q"][:, None, :],
                ],
                dim=1,
            )
            W_K_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_K"],
                    state_dict[f"blocks.{l}.attn.b_K"][:, None, :],
                ],
                dim=1,
            )

            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)

            state_dict[f"blocks.{l}.attn.W_Q"] = W_Q_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_Q"] = W_Q_eff_even[:, -1, :]
            state_dict[f"blocks.{l}.attn.W_K"] = W_K_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_K"] = W_K_eff_even[:, -1, :]

            # W_OV = W_V @ W_O
            W_V = state_dict[f"blocks.{l}.attn.W_V"]
            W_O = state_dict[f"blocks.{l}.attn.W_O"]

            # Factors the bias to be consistent.
            b_V = state_dict[f"blocks.{l}.attn.b_V"]
            b_O = state_dict[f"blocks.{l}.attn.b_O"]

            # Add singleton dimension for broadcasting
            b_V_expanded = einops.rearrange(b_V, "head_index d_head -> head_index d_head 1")

            # Element-wise multiplication of b_V and W_O
            b_V_times_W_O = b_V_expanded * W_O

            # Sum over d_head and head_index dimensions
            b_V_contribution = b_V_times_W_O.sum(1).sum(0)

            effective_bias = b_O + b_V_contribution
            state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros_like(b_V)
            state_dict[f"blocks.{l}.attn.b_O"] = effective_bias

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[f"blocks.{l}.attn.W_V"] = U @ S.diag_embed()
            state_dict[f"blocks.{l}.attn.W_O"] = utils.transpose(Vh)

        return state_dict

    def set_use_attn_result(self, use_attn_result: bool):
        """Toggle whether to explicitly calculate and expose the result for each attention head.

        Useful for interpretability but can easily burn through GPU memory.
        """
        self.cfg.use_attn_result = use_attn_result

    def set_use_split_qkv_input(self, use_split_qkv_input: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        self.cfg.use_split_qkv_input = use_split_qkv_input

    def set_use_hook_mlp_in(self, use_hook_mlp_in: bool):
        """Toggles whether to allow storing and editing inputs to each MLP layer."""

        assert not self.cfg.attn_only, "Can't use hook_mlp_in with attn_only model"
        self.cfg.use_hook_mlp_in = use_hook_mlp_in

    def set_use_attn_in(self, use_attn_in: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        assert (
            self.cfg.n_key_value_heads is None
        ), "Can't use attn_in with GroupedQueryAttention, please use split_qkv_input instead"
        self.cfg.use_attn_in = use_attn_in

    def set_ungroup_grouped_query_attention(self, ungroup_grouped_query_attention: bool):
        """
        Toggles whether to ungroup the grouped key and value heads in models with grouped query attention (GQA).
        """
        self.cfg.ungroup_grouped_query_attention = ungroup_grouped_query_attention

    def process_weights_(
        self,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ):
        """Wrapper around `load_and_process_state_dict`.

        Wrapper around load_and_process_state_dict to allow for in-place processing of the weights.
        This is useful if using HookedTransformer for training, if we then want to analyse a cleaner
        version of the same model.
        """
        state_dict = self.state_dict()
        if fold_ln and self.cfg.num_experts and self.cfg.num_experts > 1:
            # If we're using MoE, we don't fold the layer norm weights, so we don't need to do any preprocessing
            # A warning is already issued in `load_and_process_state_dict`
            pass
        elif fold_ln and self.cfg.normalization_type == "LN":
            # If we're folding the LN into the weights, we need to replace all the layernorm layers
            # with LayerNormPres, which do not have learnable parameters. This is somewhat hacky,
            # but it's the easiest way to do it.
            self.cfg.normalization_type = "LNPre"
            self.ln_final = LayerNormPre(self.cfg)
            for layer in self.blocks:
                layer.ln1 = LayerNormPre(self.cfg)
                layer.ln2 = LayerNormPre(self.cfg)
                if self.cfg.is_layer_norm_activation():
                    layer.mlp.ln = LayerNormPre(self.cfg)
        elif fold_ln and self.cfg.normalization_type == "RMS":
            # We do the same for RMSNorm if used
            self.cfg.normalization_type = "RMSPre"
            self.ln_final = RMSNormPre(self.cfg)
            for layer in self.blocks:
                layer.ln1 = RMSNormPre(self.cfg)
                layer.ln2 = RMSNormPre(self.cfg)
                if self.cfg.is_layer_norm_activation():
                    layer.mlp.ln = RMSNormPre(self.cfg)

        self.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )

    @torch.inference_mode()
    def generate(
        self,
        input: Union[str, Float[torch.Tensor, "batch pos"]] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Union[Int[torch.Tensor, "batch pos_plus_new_tokens"], str]:
        """Sample Tokens from the Model.

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish
        (by producing an EOT token), we keep running the model on the entire batch, but throw away
        the output for a finished sequence and just keep adding EOTs to pad.

        This supports entering a single string, but not a list of strings - if the strings don't
        tokenize to exactly the same length, this gets messy. If that functionality is needed,
        convert them to a batch of tokens and input that instead.

        Args:
            input (Union[str, Int[torch.Tensor, "batch pos"])]): Either a batch of tokens ([batch,
                pos]) or a text string (this will be converted to a batch of tokens with batch size
                1).
            max_new_tokens (int): Maximum number of tokens to generate.
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token.
            eos_token_id (Optional[Union[int, Sequence]]): The token ID to use for end
                of sentence. If None, use the tokenizer's eos_token_id - required if using
                stop_at_eos. It's also possible to provide a list of token IDs (not just the
                eos_token_id), in which case the generation will stop when any of them are output
                (useful e.g. for stable_lm).
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use
                greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens.
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0,
                we take the top tokens with cumulative probability >= top_p.
            temperature (float): Temperature for sampling. Higher values will make the model more
                random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is
                sampling from a uniform distribution).
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous
                tokens. Higher values will make the model more random.
            use_past_kv_cache (bool): If True, create and use cache to speed up generation.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (applicable when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos (default is True unless specified
                otherwise). Pass True or False to override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
            return_type (Optional[str]): The type of the output to return - either a string (str),
                a tensor of tokens (tensor) or whatever the format of the input was (input).
            verbose (bool): If True, show tqdm progress bars for generation.

        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
                (by default returns same type as input).
        """

        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if type(input) == str:
                # If text, convert to tokens (batch_size=1)
                assert (
                    self.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
            else:
                tokens = input

            if return_type == "input":
                if type(input) == str:
                    return_type = "str"
                else:
                    return_type = "tensor"

            assert isinstance(tokens, torch.Tensor)
            batch_size, ctx_length = tokens.shape
            device = devices.get_device_for_block_index(0, self.cfg)
            tokens = tokens.to(device)
            if use_past_kv_cache:
                past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                    self.cfg, self.cfg.device, batch_size
                )
            else:
                past_kv_cache = None

            stop_tokens: List[int] = []
            eos_token_for_padding = 0
            assert self.tokenizer is not None
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tokenizer is not None and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

                    eos_token_id = self.tokenizer.eos_token_id

                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    # eos_token_id is a Sequence (e.g. list or tuple)
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                    )

            # An array to track which sequences in the batch have finished.
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

            # Currently nothing in HookedTransformer changes with eval, but this is here in case
            # that changes in the future.
            self.eval()
            for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
                # While generating, we keep generating logits, throw away all but the final logits,
                # and then use those logits to sample from the distribution We keep adding the
                # sampled tokens to the end of tokens.
                if use_past_kv_cache:
                    # We just take the final tokens, as a [batch, 1] tensor
                    if index > 0:
                        logits = self.forward(
                            tokens[:, -1:],
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                        )
                    else:
                        logits = self.forward(
                            tokens,
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                        )
                else:
                    # We input the entire sequence, as a [batch, pos] tensor, since we aren't using
                    # the cache.
                    logits = self.forward(
                        tokens,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                    )
                final_logits = logits[:, -1, :]

                if do_sample:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=tokens,
                    ).to(devices.get_device_for_block_index(0, self.cfg))
                else:
                    sampled_tokens = final_logits.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.cfg)
                    )

                if stop_at_eos:
                    # For all unfinished sequences, add on the next token. If a sequence was
                    # finished, throw away the generated token and add eos_token_for_padding
                    # instead.
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.cfg.device),
                            torch.tensor(stop_tokens).to(self.cfg.device),
                        )
                    )

                tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

                if stop_at_eos and finished_sequences.all():
                    break

            if return_type == "str":
                if self.cfg.default_prepend_bos:
                    # If we prepended a BOS token, remove it when returning output.
                    return self.tokenizer.decode(tokens[0, 1:])
                else:
                    return self.tokenizer.decode(tokens[0])

            else:
                return tokens

    # Give access to all weights as properties.
    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """Convenience to get the unembedding matrix.

        I.e. the linear map from the final residual stream to the output logits).
        """
        return self.unembed.W_U

    @property
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        return self.unembed.b_U

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """Convenience to get the embedding matrix."""
        return self.embed.W_E

    @property
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        """Convenience function to get the positional embedding.

        Only works on models with absolute positional embeddings!
        """
        return self.pos_embed.W_pos

    @property
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        """Concatenated W_E and W_pos.

        Used as a full (overcomplete) basis of the input space, useful for full QK and full OV
        circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=0)

    # Layer-specific weights are stacked into one massive tensor and given as properties for
    # convenience and a cache is used to avoid repeated computation. Often a useful convenience when
    # we want to do analysis on weights across all layers. If GPU memory is a bottleneck, don't use
    # these properties!

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the key weights across all layers."""
        return torch.stack([block.attn.W_K for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the query weights across all layers."""
        return torch.stack([block.attn.W_Q for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the value weights across all layers."""
        return torch.stack([block.attn.W_V for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stack the attn output weights across all layers."""
        return torch.stack([block.attn.W_O for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stack the MLP input weights across all layers."""
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    def W_gate(self) -> Union[Float[torch.Tensor, "n_layers d_model d_mlp"], None]:
        """Stack the MLP gate weights across all layers.

        Only works for models with gated MLPs.
        """
        if self.cfg.gated_mlp:
            return torch.stack([block.mlp.W_gate for block in self.blocks], dim=0)
        else:
            return None

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stack the MLP output weights across all layers."""
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the key biases across all layers."""
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the query biases across all layers."""
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the value biases across all layers."""
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stack the attn output biases across all layers."""
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stack the MLP input biases across all layers."""
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stack the MLP output biases across all layers."""
        return torch.stack([block.mlp.b_out for block in self.blocks], dim=0)

    @property
    def QK(self):
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self):
        return FactoredMatrix(self.W_V, self.W_O)

    # Various utility functions
    def accumulated_bias(
        self, layer: int, mlp_input: bool = False, include_mlp_biases=True
    ) -> Float[torch.Tensor, "d_model"]:
        """Accumulated Bias.

        Returns the accumulated bias from all layer outputs (ie the b_Os and b_outs), up to the
        input of layer L.

        Args:
            layer (int): Layer number, in [0, n_layers]. layer==0 means no layers, layer==n_layers
                means all layers.
            mlp_input (bool): If True, we take the bias up to the input of the MLP
                of layer L (ie we include the bias from the attention output of the current layer,
                otherwise just biases from previous layers)
            include_mlp_biases (bool): Whether to include the biases of MLP layers. Often useful to
                have as False if we're expanding attn_out into individual heads, but keeping mlp_out
                as is.

        Returns:
            bias (torch.Tensor): [d_model], accumulated bias
        """
        accumulated_bias = torch.zeros(self.cfg.d_model, device=self.cfg.device)

        for i in range(layer):
            accumulated_bias += self.blocks[i].attn.b_O
            if include_mlp_biases:
                accumulated_bias += self.blocks[i].mlp.b_out
        if mlp_input:
            assert layer < self.cfg.n_layers, "Cannot include attn_bias from beyond the final layer"
            accumulated_bias += self.blocks[layer].attn.b_O
        return accumulated_bias

    def all_composition_scores(
        self, mode
    ) -> Float[torch.Tensor, "n_layers n_heads n_layers n_heads"]:
        """All Composition Scores.

        Returns the Composition scores for all pairs of heads, as a L1, H1, L2, H2 tensor (which is
        upper triangular on the first and third axes).

        See
        https://transformer-circuits.pub/2021/framework/index.html#:~:text=The%20above%20diagram%20shows%20Q%2D%2C%20K%2D%2C%20and%20V%2DComposition
        for three metrics used.

        Args:
            mode (str): One of ["Q", "K", "V"], the mode to use for the composition score.
        """
        left = self.OV
        if mode == "Q":
            right = self.QK
        elif mode == "K":
            right = self.QK.T
        elif mode == "V":
            right = self.OV
        else:
            raise ValueError(f"mode must be one of ['Q', 'K', 'V'] not {mode}")

        scores = utils.composition_scores(left, right, broadcast_dims=True)
        # Mask scores to be zero for all pairs with the right head in the same layer or earlier
        # layer than the left head.
        mask = (
            torch.arange(self.cfg.n_layers, device=self.cfg.device)[:, None, None, None]
            < torch.arange(self.cfg.n_layers, device=self.cfg.device)[None, None, :, None]
        )
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        return scores

    def all_head_labels(self):
        """Returns a list of all head names in the model."""
        return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]

    def load_sample_training_dataset(self, **kwargs):
        """Load Sample Training Dataset.

        Helper function to load in a 10K-20K dataset of elements from the model's training data
        distribution.

        Wrapper around utils.get_dataset, which identifies the appropriate dataset the pretrained
        models. Each dataset has a 'text' field, which contains the relevant info, some have several
        meta data fields.

        Kwargs will be passed to utils.get_dataset (e.g. cache_dir to set download location)

        Notes:

        - PT-2's training data is not open source. OpenWebText is a replication (links with
            >3 karma on Reddit)
        - OPT's training data is not open source, and is a mess of different things that is hard to
          replicate. I default to the Pile, which covers some of it, but imperfectly.

        (Some models will have actually been trained on the data supplied here, for some it's from
        the validation set).
        """
        model_dataset_map = {
            "neel": "c4_code",
            "neel-solu-old": "pile",
            "GPT2LMHeadModel": "openwebtext",
            "GPTNeoForCausalLM": "pile",
            "GPTNeoXForCausalLM": "pile",
            "GPTJForCausalLM": "pile",
            "OPTForCausalLM": "pile",
        }
        if self.cfg.original_architecture in model_dataset_map:
            self.dataset = utils.get_dataset(
                model_dataset_map[self.cfg.original_architecture], **kwargs
            )
        else:
            raise ValueError(
                f"We do not have an available dataset for the relevant model: {self.cfg.original_architecture}"
            )
        return self.dataset

    def sample_datapoint(
        self,
        tokenize: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
    ) -> Union[str, Float[torch.Tensor, "1 pos"]]:
        """Sample Data Point from Dataset.

        Helper function to randomly sample a data point from self.dataset, a small dataset from the
        data distribution the model was trained on.

        Implicitly calls self.load_sample_training_dataset if it hasn't already been called. Only
        works for pretrained models with an associated dataset. But you can manually replace
        self.dataset with a dataset of your choice if you want.

        Args:
            tokenize (bool): Whether to return tokens (instead of text). Defaults to False. Note
                that the returned tokens will be automatically truncated to the model's max context
                size.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (applicable when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos (default is True unless specified
                otherwise). Pass True or False to override the default.
            padding_side (Union[Literal["left", "right"], None], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing multiple
                strings of different lengths.
        """
        if self.dataset is None:
            self.load_sample_training_dataset()
        assert self.dataset is not None  # keep mypy happy
        sample_dataset_size = len(self.dataset)
        index = np.random.randint(0, sample_dataset_size)
        if not tokenize:
            return self.dataset[index]["text"]
        else:
            return self.to_tokens(
                self.dataset[index]["text"],
                prepend_bos=prepend_bos,
                padding_side=padding_side,
                truncate=True,
            )
