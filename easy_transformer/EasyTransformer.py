import logging
import re
from collections import namedtuple
from functools import lru_cache, partial
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm
from huggingface_hub import HfApi

# hugging face imports- useful for making this class work with a variety of popular transformers
from transformers import AutoTokenizer, PreTrainedTokenizer

import easy_transformer.loading_from_pretrained as loading
import easy_transformer.utils as utils
from easy_transformer import EasyTransformerConfig
from easy_transformer.ActivationCache import ActivationCache
from easy_transformer.components import *
from easy_transformer.FactoredMatrix import FactoredMatrix

# Note - activation cache is used with run_with_cache, past_key_value_caching is used for generation.
from easy_transformer.past_key_value_caching import EasyTransformerKeyValueCache

from . import hook_points

# Type alias for a single element tensor
Loss = TT[()]
# Named tuple object for if we want to output both logits and loss
class Output(NamedTuple):
    logits: TT["batch", "pos", "d_vocab"]
    loss: Loss


TokensTensor = TT["batch", "pos"]
InputForForwardLayer = Union[str, List[str], TokensTensor]
# TODO make this better
Internal = TT["batch", "pos", "d_model"]


class EasyTransformer(hook_points.HookedRootModule):
    """
    This class implements a full Transformer using the components in ./components.py, with
    HookPoints on every interesting activation. It inherits from HookedRootModule.

    It can have a pretrained Transformer's weights automatically loaded in via the EasyTransformer.from_pretrained class method. It can also be instantiated with randomly initialized weights via __init__ and being passed a dict or EasyTransformerConfig object.
    """

    @classmethod
    def __generate_tokenizer__(
        cls, config: EasyTransformerConfig, tokenizer: Optional[PreTrainedTokenizer]
    ):
        if tokenizer is not None:
            return tokenizer

        if config.tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            result: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name
            )
            result.eos_token = (
                result.eos_token if result.eos_token is not None else "<|endoftext|>"
            )
            result.pad_token = (
                result.pad_token if result.pad_token is not None else result.eos_token
            )
            result.bos_token = (
                result.bos_token if result.bos_token is not None else result.eos_token
            )
            return result
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens directly. In this case, we don't need a tokenizer.
            return None

    def __init_d_vocab__(self):
        if not self.cfg.d_vocab:
            # If we have a tokenizer, vocab size can be inferred from it.
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
            self.cfg.d_vocab_out = self.cfg.d_vocab

    @classmethod
    def __generate_config__(
        cls, config: Union[EasyTransformerConfig, Dict]
    ) -> EasyTransformerConfig:
        if isinstance(config, Dict):
            config = EasyTransformerConfig(**config)
            return config
        elif isinstance(config, EasyTransformerConfig):
            return config
        else:
            raise ValueError(
                "Please pass in a config dictionary or EasyTransformerConfig object. If you want to load a pretrained model, use EasyTransformer.from_pretrained() instead."
            )

    def __init_embeddings__(self):
        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.cfg)
            self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]

    def __init__normalization__(self):
        if (
            self.cfg.normalization_type == "LN"
        ):  # TODO some enum type instead of strings
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
            self.ln_final = nn.Identity()
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )

    def __init__(
        self,
        cfg,
        tokenizer=None,
        move_to_device=True,
    ):
        """
        Model initialization. Note that if you want to load the model from pretrained weights, you should use the EasyTransformer.from_pretrained() class method instead of this one.

        cfg Union[EasyTransformerConfig, Dict]: The config to use for the
            model.
        tokenizer (*optional): The tokenizer to use for the model. If not
            provided, it is inferred from cfg.tokenizer_name or initialized to None.
            If None, then the model cannot be passed strings, and d_vocab must be explicitly set.
        move_to_device (bool): Whether to move the model to the device specified in cfg.
            device.
        """
        super().__init__()
        self.cfg = EasyTransformer.__generate_config__(cfg)
        self.tokenizer = EasyTransformer.__generate_tokenizer__(self.cfg, tokenizer)
        self.__init_d_vocab__()
        self.__init_embeddings__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.cfg, block_index)
                for block_index in range(self.cfg.n_layers)
            ]
        )

        self.__init__normalization__()

        self.unembed = Unembed(self.cfg)

        # TODO rename [cfg] to [config]- would require a bunch of rewrites and some good testing
        if self.cfg.init_weights:
            self.init_weights()

        if move_to_device:
            self.to(self.cfg.device)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    def __make_tokens_for_forward__(
        self, input: InputForForwardLayer, prepend_bos: bool
    ) -> TokensTensor:
        tokens: torch.Tensor  # set inside the function body
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # This is only intended to support passing in a single string
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            assert isinstance(
                input, torch.Tensor
            )  # typecast; we know that this is ok because of the above logic
            tokens = input
        if len(tokens.shape) == 1:
            # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
            tokens = tokens[None]
        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(self.cfg.device)
        assert isinstance(tokens, torch.Tensor)
        return tokens

    def __get_pos_offset_for_forward__(
        self,
        tokens: TokensTensor,
        past_kv_cache: Optional[EasyTransformerKeyValueCache],
    ):
        # If we're doing caching, then we reuse keys and values from previous runs, as that's the only
        # way that past activations will affect the final logits. The cache contains those so we don't
        # need to recompute them. This is useful for generating text. As we have absolute positional
        # encodings, to implement this we have a `pos_offset` variable, defaulting to zero, which says
        # to offset which positional encodings are used (cached keys and values were calculated with
        # their own positional encodings).
        if past_kv_cache is None:
            return 0
        else:
            batch_size, ctx_length = tokens.shape
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            assert num_heads_in_cache == self.cfg.n_heads
            assert d_head_in_cache == self.cfg.d_head
            # If we want to generate from the empty string, we'd pass in an empty cache, so we need to handle that case
            assert (
                cache_ctx_length == 0 or ctx_length == 1
            ), "Pass in one token at a time after loading cache"
            return cache_ctx_length

    def __get_embedding__(
        self, tokens: TokensTensor, pos_offset: int
    ) -> Tuple[Internal, Optional[Internal]]:
        embed: Internal = self.hook_embed(self.embed(tokens))
        if self.cfg.positional_embedding_type == "standard":
            pos_embed: TT["batch", "pos", "d_model"] = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset)
            )
            return embed + pos_embed, None
        elif self.cfg.positional_embedding_type == "shortformer":
            # If we're using shortformer style attention, we don't add the positional embedding to the residual stream. See EasyTransformerConfig for details
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset)
            )  # [batch, pos, d_model]
            return embed, pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            # Rotary doesn't use positional embeddings, instead they're applied when dot producting keys and queries. See EasyTransformerConfig for details
            return embed, None
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )

    def __handle_return_for_forward__(
        self, residual: Internal, tokens: TokensTensor, return_type: Optional[str]
    ):
        if return_type is None:
            return None
        else:
            logits = self.unembed(residual)  # [batch, pos, d_vocab]
            if return_type == "logits":
                return logits
            else:
                assert (
                    not self.cfg.bert_family
                ), "BERT-family models do not have a supported loss function"
                loss = lm_cross_entropy_loss(logits, tokens)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return (logits, loss)
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None

    # TODO make sure type assertions are provided
    def forward(
        self,
        input: InputForForwardLayer,
        return_type: Optional[str] = "logits",
        prepend_bos: bool = True,
        past_kv_cache: Optional[EasyTransformerKeyValueCache] = None,
    ) -> Union[
        None,
        TT["batch", "pos", "d_vocab"],
        Loss,
        Tuple[TT["batch", "pos", "d_vocab"], Loss],
    ]:
        """Input is either a batch of tokens ([batch, pos]) or a text string, a string is automatically tokenized to a batch of a single element. The prepend_bos flag only applies when inputting a text string.

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), 'logits' (return logits), 'loss' (return cross-entropy loss), 'both' (return logits and loss)
        prepend_bos bool: Whether to prepend the BOS token to the input. Only applies when input is a string. Defaults to True (unlike to_tokens) - even for models not explicitly trained with this, heads often use the first position as a resting position and accordingly lose information from the first token, so this empirically seems to give better results.

        Note that loss is the standard "predict the next token" cross-entropy loss for GPT-2 style language models - if you want a custom loss function, the recommended behaviour is returning the logits and then applying your custom loss function.
        """
        tokens = self.__make_tokens_for_forward__(input=input, prepend_bos=prepend_bos)
        pos_offset = self.__get_pos_offset_for_forward__(
            tokens=tokens, past_kv_cache=past_kv_cache
        )
        embedding = self.__get_embedding__(tokens=tokens, pos_offset=pos_offset)

        # Cache is contains a list of EasyTransformerKeyValueCache objects, one for each block
        get_past_kv_cache_entry = (
            lambda i: past_kv_cache[i] if past_kv_cache is not None else None
        )

        residual, shortformer_pos_embed = embedding
        for i, block in enumerate(self.blocks):
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(
                residual,
                past_kv_cache_entry=get_past_kv_cache_entry(i),
                shortformer_pos_embed=shortformer_pos_embed,
            )
        if self.cfg.normalization_type is not None:
            residual = self.ln_final(residual)
        return self.__handle_return_for_forward__(
            return_type=return_type, residual=residual, tokens=tokens
        )

    def loss_fn(
        self,
        logits: TT["batch", "pos", "d_vocab"],
        tokens: TT["batch", "pos"],
        per_token: bool = False,
    ):
        """
        Wrapper around utils.lm_cross_entropy_loss, used in forward() with return_type=="loss" or "both".
        """
        return utils.lm_cross_entropy_loss(logits, tokens, per_token)

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            TT["batch", "pos", "d_vocab"],
            Loss,
            Tuple[TT["batch", "pos", "d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful EasyTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
            )
            return out, cache
        else:
            return out, cache_dict

    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """
        Sets the tokenizer to use for this model.
        tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer
        """
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: bool = True,
        move_to_device: bool = True,
    ) -> TT["batch", "pos"]:  # TODO change this type
        """
        Converts a string to a tensor of tokens. If prepend_bos is True, prepends the BOS token to the input - this is recommended when creating a sequence of tokens to be input to a model.

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when inputting a prompt to the model as the first token is often treated weirdly, but should only be done at the START of the prompt. Make sure to turn it off if you're looking at the tokenization of part of the prompt!
        (Note: some models eg GPT-2 were not trained with a BOS token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not careful!
        """
        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
        if prepend_bos:
            if isinstance(input, str):
                input = self.tokenizer.bos_token + input
            else:
                input = [self.tokenizer.bos_token + string for string in input]
        tokens = self.tokenizer(input, return_tensors="pt", padding=True)["input_ids"]
        if move_to_device:
            tokens = tokens.to(self.cfg.device)
        return tokens

    def to_string(
        self, tokens: Union[TT["batch", "pos"], TT["pos"], np.ndarray, List[TT["pos"]]]
    ) -> Union[str, List[str]]:
        """
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
            return self.tokenizer.batch_decode(
                tokens, clean_up_tokenization_spaces=False
            )
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passed in: {tokens.shape}")

    def to_str_tokens(
        self,
        input: Union[str, Union[TT["pos"], TT[1, "pos"]], list],
        prepend_bos: bool = True,
    ) -> List[str]:
        """Method to map text, a list of text or tokens to a list of tokens as strings

        Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when inputting a prompt to the model as the first token is often treated weirdly, but should only be done at the START of the prompt. Make sure to turn it off if you're looking at the tokenization of part of the prompt!
        (Note: some models eg GPT-2 were not trained with a BOS token, others (OPT and my models) were)

        Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not careful!

        Args:
            input (Union[str, list, torch.Tensor]): The input - either a string or a tensor of tokens. If tokens, should be a tensor of shape [pos] or [1, pos]
            prepend_bos (bool, optional): Whether to prepend a BOS token. Only applies if input is a string. Defaults to True.

        Returns:
            str_tokens: List of individual tokens as strings
        """
        if isinstance(input, list):
            return list(
                map(lambda tokens: self.to_str_tokens(tokens, prepend_bos), input)
            )  # type: ignore
        elif isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)[0]
        elif isinstance(input, torch.Tensor):
            tokens = input
            tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
            assert (
                tokens.dim() == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        elif isinstance(input, np.ndarray):
            tokens = input
            tokens = tokens.squeeze()  # Get rid of a trivial batch dimension
            assert (
                tokens.ndim == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        else:
            raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
        str_tokens = self.tokenizer.batch_decode(
            tokens, clean_up_tokenization_spaces=False
        )
        return str_tokens

    def to_single_token(self, string):
        """Maps a string that makes up a single token to the id for that token. Raises an error for strings that are not a single token! If uncertain use to_tokens"""

        # We use the to_tokens method, do not append a BOS token
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        # If token shape is non-empty, raise error
        assert not token.shape, f"Input string: {string} is not a single token!"
        return token.item()

    def get_token_position(
        self,
        single_token: Union[str, int],
        tokens: Union[str, Union[TT["pos"], TT[1, "pos"]]],
        mode="first",
        prepend_bos=True,
    ):
        """
        Get the position of a single_token in a sequence of tokens.

        Gotcha: If you're inputting a string, it'll automatically be tokenized. Be careful about prepend_bos is true or false! When a string is input to the model, a BOS (beginning of sequence) token is prepended by default when the string is tokenized. But this should only be done at the START of the input, not when inputting part of the prompt. If you're getting weird off-by-one errors, check carefully for what the setting should be!

        Args:
            single_token (Union[str, int]): The token to search for. Can
                be a token index, or a string (but the string must correspond to a
                single token)
            tokens (Union[str, torch.Tensor]): The sequence of tokens to
                search in. Can be a string or a rank 1 tensor or a rank 2 tensor
                with a dummy batch dimension.
            mode (str, optional): If there are multiple matches, which match to return. Supports "first" or "last". Defaults to "first".
            prepend_bos (bool): Prepends a BOS (beginning of sequence) token when tokenizing a string. Only matters when inputting a string to
                the function, otherwise ignored.
        """
        if isinstance(tokens, str):
            # If the tokens are a string, convert to tensor
            tokens = self.to_tokens(tokens, prepend_bos=prepend_bos)

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

        indices = torch.arange(len(tokens))[tokens == single_token]
        assert len(indices) > 0, f"The token does not occur in the prompt"
        if mode == "first":
            return indices[0].item()
        elif mode == "last":
            return indices[-1].item()
        else:
            raise ValueError(f"mode must be 'first' or 'last', not {mode}")

    def tokens_to_residual_directions(
        self, tokens: Union[str, int, TT[()], TT["position"], TT["batch", "position"]]
    ) -> Union[
        TT["d_model"], TT["position", "d_model"], TT["batch", "position", "d_model"]
    ]:
        """Maps tokens to a tensor with the unembedding vector for those tokens, ie the vector in the residual stream that we dot with to the get the logit for that token.

        WARNING: If you use this without folding in LayerNorm, the results will be misleading and may be incorrect, as the LN weights change the unembed map. This is done automatically with the fold_ln flag on from_pretrained

        WARNING 2: LayerNorm scaling will scale up or down the effective direction in the residual stream for each output token on any given input token position. ActivationCache.apply_ln_to_stack will apply the appropriate scaling to these directions.

        Args:
            tokens (Union[str, int, torch.Tensor]): The token(s). If a single token, can be a single element tensor, an integer, or string. If string, will be mapped to a single token using to_single_token, and an error raised if it's multiple tokens.
            The method also works for a batch of input tokens

        Returns:
            residual_direction torch.Tensor: The unembedding vector for the token(s), a stack of [d_model] tensor.
        """
        if isinstance(tokens, torch.Tensor) and tokens.numel() > 1:
            # If the tokens are a tensor, and have more than one element, assume they are a batch of tokens
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

    def to(self, device_or_dtype):
        """
        Wrapper around to that also changes self.cfg.device if it's a torch.device or string. If torch.dtype, just passes through
        """
        if isinstance(device_or_dtype, torch.device):
            self.cfg.device = device_or_dtype.type
            logging.info("Moving model to device: ", self.cfg.device)
        elif isinstance(device_or_dtype, str):
            self.cfg.device = device_or_dtype
            logging.info("Moving model to device: ", self.cfg.device)
        elif isinstance(device_or_dtype, torch.dtype):
            logging.info("Changing model dtype to", device_or_dtype)
        return nn.Module.to(self, device_or_dtype)

    def cuda(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cuda")

    def cpu(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cpu")

    # written in this style because this guarantees that [self] isn't set inside here
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        refactor_factored_attn_matrices=False,
        checkpoint_index=None,
        checkpoint_value=None,
        hf_model=None,
        device=None,
        move_state_dict_to_device=True,
        **model_kwargs,
    ):
        """Class method to load in a pretrained model weights to the EasyTransformer format and optionally to do some processing to make the model easier to interpret. Currently supports loading from most autoregressive HuggingFace models (GPT2, GPTNeo, GPTJ, OPT) and from a range of toy models and SoLU models trained by me (Neel Nanda).

        Also supports loading from a checkpoint for checkpointed models (currently, models trained by me (NeelNanda) and the stanford-crfm models). These can either be determined by the checkpoint index (the index of the checkpoint in the checkpoint list) or by the checkpoint value (the value of the checkpoint, eg 1000 for a checkpoint taken at step 1000 or after 1000 tokens. Each model has checkpoints labelled with exactly one of labels and steps). If neither is specified the final model is loaded. If both are specified, the checkpoint index is used.

        See load_and_process_state_dict for details on the processing (folding layer norm, centering the unembedding and centering the writing weights)

        Args:
            model_name (str): The model name - must be an element of OFFICIAL_MODEL_NAMES or an alias of one.
            fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the
                subsequent linear layer. This does not change the computation.
                Defaults to True.
            center_writing_weights (bool, optional): Whether to center weights
            writing to
                the residual stream (ie set mean to be zero). Due to LayerNorm
                this doesn't change the computation. Defaults to True.
            center_unembed (bool, optional): Whether to center W_U (ie set mean
            to be zero).
                Softmax is translation invariant so this doesn't affect log
                probs or loss, but does change logits. Defaults to True.
            refactor_factored_attn_matrices (bool, optional): Whether to convert the factored
                matrices (W_Q & W_K, and W_O & W_V) to be "even". Defaults to False
            checkpoint_index (int, optional): If loading from a checkpoint, the index of
                the checkpoint to load. Defaults to None.
            checkpoint_value (int, optional): If loading from a checkpoint, the value of
                the checkpoint to load, ie the step or token number (each model
                has checkpoints labelled with exactly one of these). Defaults to
                None.
            hf_model (AutoModelForCausalLM, optional): If you have already loaded in the
                HuggingFace model, you can pass it in here rather than needing
                to recreate the object. Defaults to None.
            device (str, optional): The device to load the model onto. By
                default will load to CUDA if available, else CPU.
            move_state_dict_to_device (bool): Whether to move the state dict to the
                relevant device before processing and loading in the weights.
                Defaults to True.
            model_kwargs (dict, optional): Any additional kwargs to pass to the
                EasyTransformer initialization.
        """
        logging.info(f"Loading model: {model_name}")

        # Get the model name used in HuggingFace, rather than the alias.
        official_model_name = loading.get_official_model_name(model_name)

        # Load the config into an EasyTransformerConfig object If loading from a
        # checkpoint, the config object will contain the information about the
        # checkpoint
        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=fold_ln,
            device=device,
        )

        # Get the state dict of the model (ie a mapping of parameter names to tensors), processed to match the EasyTransformer parameter names.
        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model
        )

        # Create the EasyTransformer object
        model = cls(cfg, **model_kwargs)

        model.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            move_state_dict_to_device=move_state_dict_to_device,
        )

        logging.info(
            f"Finished loading pretrained model {model_name} into EasyTransformer!"
        )

        return model

    def init_weights(self):
        """
        Initialize weights matrices with a normal of std=initializer_range (default=0.02). This roughly follows the GPT-2 paper's scheme (but with truncation, and not halving the std for W_pos).

        LayerNorm weights are already initialized to 1.0, and all biases are initialized to 0.0 (including LayerNorm), so this just initializes weight matrices.

        Weight matrices are set to empty by default (to save space + compute, since they're the bulk of the parameters), so it is important to call this if you are not loading in pretrained weights! Note that this function assumes that weight names being with W_

        Set seed here to ensure determinism.

        This does NOT follow the PyTorch scheme, which as far as I can tell is super out of date but no one has gotten round to updating it?
        https://github.com/pytorch/pytorch/issues/18182

        PyTorch Transformers are especially bad - TransformerEncoder initializes all layers to the exact same weights?! https://github.com/pytorch/pytorch/issues/72253

        The best paper I've found on transformer initialization is the muP paper, but haven't integrated those ideas yet: https://arxiv.org/abs/2203.03466
        """

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=self.cfg.initializer_range)

    def load_and_process_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        refactor_factored_attn_matrices: bool = False,
        move_state_dict_to_device: bool = True,
    ):
        """Method to load a state dict into the model, and to apply processing to simplify it. The state dict is assumed to be in the EasyTransformer format.

        See the relevant method (same name as the flag) for more details on the folding, centering and processing flags.

        Args:
            state_dict (dict): The state dict of the model, in EasyTransformer format
            fold_ln (bool, optional): Whether to fold in the LayerNorm weights to the
                subsequent linear layer. This does not change the computation. Defaults to True.
            center_writing_weights (bool, optional): Whether to center weights writing to the
                residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the computation. Defaults to True.
            center_unembed (bool, optional): Whether to center W_U (ie set mean to be zero).
                Softmax is translation invariant so this doesn't affect log probs or loss, but does change logits. Defaults to True.
            refactor_factored_attn_matrices (bool, optional): Whether to convert the factored
                matrices (W_Q & W_K, and W_O & W_V) to be "even". Defaults to False
            move_state_dict_to_device (bool, optional): Whether to move the state dict to the device of the model. Defaults to True.
        """

        if move_state_dict_to_device:
            state_dict = {k: v.to(self.cfg.device) for k, v in state_dict.items()}
        state_dict = self.fill_missing_keys(state_dict)
        if fold_ln:
            if self.cfg.normalization_type not in ["LN", "LNPre"]:
                logging.warning(
                    "You are not using LayerNorm, so the layer norm weights can't be folded! Skipping"
                )
            else:
                # Note - you can run fold_layer_norm while normalization_type is LN, but this is not advised! It mostly goes wrong when you're training the model.
                state_dict = self.fold_layer_norm(state_dict)
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
        if refactor_factored_attn_matrices:
            state_dict = self.refactor_factored_attn_matrices(state_dict)
        self.load_state_dict(state_dict)

    def fill_missing_keys(self, state_dict):
        """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.

        This function is assumed to be run before weights are initialized.

        Args:
            state_dict (dict): State dict from a pretrained model

        Returns:
            dict: State dict with missing keys filled in
        """
        # Get the default state dict
        default_state_dict = self.state_dict()
        # Get the keys that are missing from the pretrained model
        missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
        # Fill in the missing keys with the default initialization
        for key in missing_keys:
            if "hf_model" in key:
                # Skip keys that are from the HuggingFace model, if loading from HF.
                continue
            if "W_" in key:
                logging.warning(
                    "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                        key
                    )
                )
            state_dict[key] = default_state_dict[key]
        return state_dict

    def fold_layer_norm(self, state_dict: Dict[str, torch.Tensor]):
        """Takes in a state dict from a pretrained model, formatted to be consistent with EasyTransformer but with LayerNorm weights and biases. Folds these into the neighbouring weights. See EasyTransformerConfig for more details

        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model
        """
        for l in range(self.cfg.n_layers):
            # Fold ln1 into attention - it's important to fold biases first,
            # since biases depend on weights but not vice versa
            # The various indexing is just to broadcast ln.b and ln.w along every axis other than d_model. Each weight matrix right multiplies.
            # To fold in the bias, we use the W_ matrix to map it to the hidden space of the layer, so we need to sum along axis -2, which is the residual stream space axis.
            state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[f"blocks.{l}.attn.b_Q"] + (
                state_dict[f"blocks.{l}.attn.W_Q"]
                * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
            ).sum(-2)
            state_dict[f"blocks.{l}.attn.b_K"] = state_dict[f"blocks.{l}.attn.b_K"] + (
                state_dict[f"blocks.{l}.attn.W_K"]
                * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
            ).sum(-2)
            state_dict[f"blocks.{l}.attn.b_V"] = state_dict[f"blocks.{l}.attn.b_V"] + (
                state_dict[f"blocks.{l}.attn.W_V"]
                * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
            ).sum(-2)

            state_dict[f"blocks.{l}.attn.W_Q"] = (
                state_dict[f"blocks.{l}.attn.W_Q"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.W_K"] = (
                state_dict[f"blocks.{l}.attn.W_K"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.W_V"] = (
                state_dict[f"blocks.{l}.attn.W_V"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            del (
                state_dict[f"blocks.{l}.ln1.w"],
                state_dict[f"blocks.{l}.ln1.b"],
            )

            # Fold ln2 into MLP
            if not self.cfg.attn_only:
                state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[
                    f"blocks.{l}.mlp.b_in"
                ] + (
                    state_dict[f"blocks.{l}.mlp.W_in"]
                    * state_dict[f"blocks.{l}.ln2.b"][:, None]
                ).sum(
                    -2
                )
                state_dict[f"blocks.{l}.mlp.W_in"] = (
                    state_dict[f"blocks.{l}.mlp.W_in"]
                    * state_dict[f"blocks.{l}.ln2.w"][:, None]
                )
                del state_dict[f"blocks.{l}.ln2.w"], state_dict[f"blocks.{l}.ln2.b"]

                if self.cfg.act_fn.startswith("solu"):
                    # Fold ln3 into activation
                    state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[
                        f"blocks.{l}.mlp.b_out"
                    ] + (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.b"][:, None]
                    ).sum(
                        -2
                    )
                    state_dict[f"blocks.{l}.mlp.W_out"] = (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.w"][:, None]
                    )
                    del (
                        state_dict[f"blocks.{l}.mlp.ln.w"],
                        state_dict[f"blocks.{l}.mlp.ln.b"],
                    )
        # Fold ln_final into Unembed
        if not self.cfg.final_rms:
            # Dumb bug from my old SoLU training code, some models have RMSNorm instead of LayerNorm pre unembed.
            state_dict[f"unembed.b_U"] = state_dict[f"unembed.b_U"] + (
                state_dict[f"unembed.W_U"] * state_dict[f"ln_final.b"][:, None]
            ).sum(dim=-2)
            del state_dict[f"ln_final.b"]
        state_dict[f"unembed.W_U"] = (
            state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"][:, None]
        )
        del state_dict[f"ln_final.w"]
        return state_dict

    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and W_out. This is done by subtracting the mean of the weights from the weights themselves. This is done in-place. See fold_layer_norm for more details."""
        state_dict["embed.W_E"] = state_dict["embed.W_E"] - state_dict[
            "embed.W_E"
        ].mean(-1, keepdim=True)
        if self.cfg.positional_embedding_type != "rotary":
            state_dict["pos_embed.W_pos"] = state_dict["pos_embed.W_pos"] - state_dict[
                "pos_embed.W_pos"
            ].mean(-1, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f"blocks.{l}.attn.W_O"] = state_dict[
                f"blocks.{l}.attn.W_O"
            ] - state_dict[f"blocks.{l}.attn.W_O"].mean(
                -1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[f"blocks.{l}.attn.b_O"] = (
                state_dict[f"blocks.{l}.attn.b_O"]
                - state_dict[f"blocks.{l}.attn.b_O"].mean()
            )  # b_O is [d_model]
            if not self.cfg.attn_only:
                state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[
                    f"blocks.{l}.mlp.W_out"
                ] - state_dict[f"blocks.{l}.mlp.W_out"].mean(-1, keepdim=True)
                state_dict[f"blocks.{l}.mlp.b_out"] = (
                    state_dict[f"blocks.{l}.mlp.b_out"]
                    - state_dict[f"blocks.{l}.mlp.b_out"].mean()
                )
        return state_dict

    def center_unembed(self, state_dict: Dict[str, torch.Tensor]):
        """Centers the unembedding weights W_U. This is done by subtracting the mean of the weights from the weights themselves. This is done in-place. As softmax is translation invariant, this changes the logits but not the log probs, and makes the model logits (slightly) more interpretable - when trying to understand how components contribute to the logits, we'll be less misled by components that just add something to every logit."""
        state_dict["unembed.W_U"] = state_dict["unembed.W_U"] - state_dict[
            "unembed.W_U"
        ].mean(-1, keepdim=True)
        state_dict["unembed.b_U"] = (
            state_dict["unembed.b_U"] - state_dict["unembed.b_U"].mean()
        )
        return state_dict

    def refactor_factored_attn_matrices(self, state_dict: Dict[str, torch.Tensor]):
        """
        Experimental method for managing queries, keys and values. As argued in [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), queries, keys and values are somewhat arbitrary intermediate terms when computing with the low rank factored matrices W_QK = W_Q @ W_K.T and W_OV = W_V @ W_O, and these matrices are the only thing determining head behaviour. But there are many ways to find a low rank factorization to a given matrix, and hopefully some of these are more interpretable than others! This method is one attempt, which makes all of the matrices have orthogonal rows or columns, W_O into a rotation and W_Q and W_K having the nth column in each having the same norm. The formula is $W_V = U @ S,W_O=Vh.T,W_Q=U@S.sqrt(),W_K=Vh@S.sqrt()$.

        More details:

        If W_OV = U @ S @ Vh.T in its singular value decomposition, (where S is in R^d_head not R^d_model, as W_OV is low rank), W_OV = (U @ S) @ (Vh.T) is an equivalent low rank factorisation, where rows/columns of each matrix are orthogonal! So setting $W_V=US$ and $W_O=Vh.T$ works just as well. I *think* this is a more interpretable setup, because now $W_O$ is just a rotation, and doesn't change the norm, so $z$ has the same norm as the result of the head.

        For $W_QK = W_Q @ W_K.T$ we use the refactor $W_Q = U @ S.sqrt()$ and $W_K = Vh @ S.sqrt()$, which is also equivalent ($S==S.sqrt() @ S.sqrt()$ as $S$ is diagonal). Here we keep the matrices as having the same norm, since there's not an obvious asymmetry between the keys and queries.

        Biases are more fiddly to deal with. For OV it's pretty easy - we just need (x @ W_V + b_V) @ W_O + b_O to be preserved, so we can set b_V' = 0. and b_O' = b_V @ W_O + b_O (note that b_V in R^{head_index x d_head} while b_O in R^{d_model}, so we need to sum b_V @ W_O along the head_index dimension too).

        For QK it's messy - we need to preserve the bilinear form of (x @ W_Q +
        b_Q) * (y @ W_K + b_K), which is fairly messy. To deal with the biases,
        we concatenate them to W_Q and W_K to simulate a d_model+1 dimensional
        input (whose final coordinate is always 1), do the SVD factorization on
        this effective matrix, then separate out into final weights and biases


        """
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
            effective_bias = b_O + einsum(
                "head_index d_head, head_index d_head d_model -> d_model", b_V, W_O
            )
            state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros_like(b_V)
            state_dict[f"blocks.{l}.attn.b_O"] = effective_bias

            # Helper class to efficiently deal with low rank factored matrices.
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[f"blocks.{l}.attn.W_V"] = U @ S.diag_embed()
            state_dict[f"blocks.{l}.attn.W_O"] = utils.transpose(Vh)

        return state_dict

    def set_use_attn_result(self, use_attn_result):
        """
        Toggles whether to explicitly calculate and expose the result for each attention head - useful for interpretability but can easily burn through GPU memory.
        """
        self.cfg.use_attn_result = use_attn_result

    def process_weights_(
        self,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        refactor_factored_attn_matrices: bool = False,
        move_state_dict_to_device: bool = True,
    ):
        """
        Wrapper around load_and_process_state_dict to allow for in-place processing of the weights. This is useful if using EasyTransformer for training, if we then want to analyse a cleaner version of the same model.
        """
        state_dict = self.state_dict()
        if fold_ln and self.cfg.normalization_type == "LN":
            # If we're folding the LN into the weights, we need to replace all of the layernorm layers with LayerNormPres, which do not have learnable parameters.
            # This is somewhat hacky, but it's the easiest way to do it.
            self.cfg.normalization_type = "LNPre"
            self.ln_final = LayerNormPre(self.cfg)
            for layer in self.blocks:
                layer.ln1 = LayerNormPre(self.cfg)
                layer.ln2 = LayerNormPre(self.cfg)
                if self.cfg.act_fn.endswith("_ln"):
                    layer.mlp.ln = LayerNormPre(self.cfg)

        self.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            move_state_dict_to_device=move_state_dict_to_device,
        )

    @torch.inference_mode()
    def generate(
        self,
        input: Union[str, TT["batch", "pos"]] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        num_return_sequences: int = 1,
        use_past_kv_cache: bool = True,
        prepend_bos=True,
        return_type: Optional[str] = "input",
    ) -> TT["batch", "pos + new_tokens"]:
        """
        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.

        To avoid fiddling with ragged tensors, if we input a batch of text and some sequences finish (by producing an EOT token), we keep running the model on the entire batch, but throw away the output for a finished sequence and just keep adding EOTs to pad.

        This supports entering a single string, but not a list of strings - if the strings don't tokenize to exactly the same length, this gets messy. If that functionality is needed, convert them to a batch of tokens and input that instead.

        Args:
            input (int): Either a batch of tokens ([batch, pos]) or a text string (this will be converted to a batch of tokens with batch size 1)
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use greedy search (take the max logit each time).
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens. If <1.0, we take the top tokens with cumulative probability >= top_p
            temperature (float): Temperature for sampling. Higher values will make the model more random (limit of temp -> 0 is just taking the top token, limit of temp -> inf is sampling from a uniform distribution)
            freq_penalty (float): Frequency penalty for sampling - how much to penalise previous tokens. Higher values will make the model more random
            use_past_kv_cache (bool): If True, create and use cache to speed up generation
            prepend_bos (bool): If True, prepend the model's bos_token_id to the input, if it's a string. Irrelevant if input is a tensor.
            return_type (str, *optional*): The type of the output to return - either a string (str), a tensor of tokens (tensor) or whatever the format of the input was (input).
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens - by default returns same type as input
        """
        if type(input) == str:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input

        if return_type == "input":
            if type(input) == str:
                return_type = "str"
            else:
                return_type = "tensor"

        assert isinstance(tokens, torch.Tensor)
        batch_size, ctx_length = tokens.shape
        tokens = tokens.to(self.cfg.device)
        if use_past_kv_cache:
            past_kv_cache = EasyTransformerKeyValueCache.init_cache(
                self.cfg, self.cfg.device, batch_size
            )
        else:
            past_kv_cache = None

        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"

            eos_token_id = self.tokenizer.eos_token_id

        # An array to track which sequences in the batch have finished.
        finished_sequences = torch.zeros(
            batch_size, dtype=torch.bool, device=self.cfg.device
        )

        # Currently nothing in EasyTransformer changes with eval, but this is here in case that changes in the future
        self.eval()
        for index in tqdm.tqdm(range(max_new_tokens)):
            # While generating, we keep generating logits, throw away all but the final logits, and then use those logits to sample from the distribution
            # We keep adding the sampled tokens to the end of tokens.
            if use_past_kv_cache:
                # We just take the final tokens, as a [batch, 1] tensor
                if index > 0:
                    logits = self.forward(
                        tokens[:, -1:],
                        return_type="logits",
                        past_kv_cache=past_kv_cache,
                    )
                else:
                    logits = self.forward(
                        tokens, return_type="logits", past_kv_cache=past_kv_cache
                    )

            else:
                # We input the entire sequence, as a [batch, pos] tensor, since we aren't using the cache
                logits = self.forward(tokens, return_type="logits")
            final_logits = logits[:, -1, :]

            sampled_tokens = utils.sample_logits(
                final_logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
                tokens=tokens,
            )

            if stop_at_eos:
                # For all unfinished sequences, add on the next token. If a sequence finished, we throw away the generated token and instead add an EOS token to pad.
                sampled_tokens[finished_sequences] = eos_token_id
                finished_sequences.logical_or_(sampled_tokens == eos_token_id)

            tokens = torch.cat([tokens, sampled_tokens.unsqueeze(-1)], dim=-1)

            if stop_at_eos and finished_sequences.all():
                break

        if return_type == "str":
            if prepend_bos:
                # If we prepended a BOS token, remove it when returning output.
                return self.tokenizer.decode(tokens[0, 1:])
            else:
                return self.tokenizer.decode(tokens[0])

        else:
            return tokens

    # Give access to all weights as properties.
    @property
    def W_U(self) -> TT["d_model", "d_vocab"]:
        """
        Convenience to get the unembedding matrix (ie the linear map from the final residual stream to the output logits)
        """
        return self.unembed.W_U

    @property
    def b_U(self) -> TT["d_vocab"]:
        return self.unembed.b_U

    @property
    def W_E(self) -> TT["d_vocab", "d_model"]:
        """
        Convenience to get the embedding matrix
        """
        return self.embed.W_E

    @property
    def W_pos(self) -> TT["n_ctx", "d_model"]:
        """
        Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
        """
        return self.pos_embed.W_pos

    @property
    def W_E_pos(self) -> TT["d_vocab + n_ctx", "d_model"]:
        """
        Concatenated W_E and W_pos. Used as a full (overcomplete) basis of the input space, useful for full QK and full OV circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=1)

    # Layer-specific weights are stacked into one massive tensor and given as properties for convenience and a cache is used to avoid repeated computation. Often a useful convenience when we want to do analysis on weights across all layers. If GPU memory is a bottleneck, don't use these properties!

    @property
    @lru_cache(maxsize=None)
    def W_K(self) -> TT["n_layers", "n_heads", "d_model", "d_head"]:
        """Stacks the key weights across all layers"""
        return torch.stack([block.attn.W_K for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def W_Q(self) -> TT["n_layers", "n_heads", "d_model", "d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack([block.attn.W_Q for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def W_V(self) -> TT["n_layers", "n_heads", "d_model", "d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack([block.attn.W_V for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def W_O(self) -> TT["n_layers", "n_heads", "d_head", "d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack([block.attn.W_O for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def W_in(self) -> TT["n_layers", "d_model", "d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def W_out(self) -> TT["n_layers", "d_mlp", "d_model"]:
        """Stacks the MLP output weights across all layers"""
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def b_K(self) -> TT["n_layers", "n_heads", "d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def b_Q(self) -> TT["n_layers", "n_heads", "d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def b_V(self) -> TT["n_layers", "n_heads", "d_head"]:
        """Stacks the value biases across all layers"""
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def b_O(self) -> TT["n_layers", "d_model"]:
        """Stacks the attn output biases across all layers"""
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def b_in(self) -> TT["n_layers", "d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    @lru_cache(maxsize=None)
    def b_out(self) -> TT["n_layers", "d_model"]:
        """Stacks the MLP output biases across all layers"""
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
    ) -> TT["layers_accumulated_over", "d_model"]:
        """Returns the accumulated bias from all layer outputs (ie the b_Os and b_outs), up to the input of layer L.

        Args:
            layer (int): Layer number, in [0, n_layers]. layer==0 means no layers, layer==n_layers means all layers.
            mlp_input (bool): If True, we take the bias up to the input of the MLP of layer L (ie we include the bias from the attention output of the current layer, otherwise just biases from previous layers)
            include_mlp_biases (bool): Whether to include the biases of MLP layers. Often useful to have as False if we're expanding attn_out into individual heads, but keeping mlp_out as is.
        Returns:
            bias (torch.Tensor): [d_model], accumulated bias
        """
        accumulated_bias = torch.zeros(self.cfg.d_model, device=self.cfg.device)

        for i in range(layer):
            accumulated_bias += self.blocks[i].attn.b_O
            if include_mlp_biases:
                accumulated_bias += self.blocks[i].mlp.b_out
        if mlp_input:
            assert (
                layer < self.cfg.n_layers
            ), "Cannot include attn_bias from beyond the final layer"
            accumulated_bias += self.blocks[layer].attn.b_O
        return accumulated_bias

    def all_composition_scores(
        self, mode
    ) -> TT["n_layers", "n_heads", "n_layers", "n_heads"]:
        """Returns the Composition scores for all pairs of heads, as a L1, H1, L2, H2 tensor (which is upper triangular on the first and third axes)

        mode is one of ["Q", "K", "V"]

        See https://transformer-circuits.pub/2021/framework/index.html#:~:text=The%20above%20diagram%20shows%20Q%2D%2C%20K%2D%2C%20and%20V%2DComposition for three metrics used
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
        # Mask scores to be zero for all pairs with the right head in the same layer or earlier layer than the left head.
        mask = (
            torch.arange(self.cfg.n_layers, device=self.cfg.device)[:, None, None, None]
            < torch.arange(self.cfg.n_layers, device=self.cfg.device)[
                None, None, :, None
            ]
        )
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        return scores

    def all_head_labels(self):
        return [
            f"L{l}H{h}"
            for l in range(self.cfg.n_layers)
            for h in range(self.cfg.n_heads)
        ]
