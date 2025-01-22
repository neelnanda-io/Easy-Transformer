"""Activation Cache.

The :class:`ActivationCache` is at the core of Transformer Lens. It is a wrapper that stores all
important activations from a forward pass of the model, and provides a variety of helper functions
to investigate them.

Getting Started:

When reading these docs for the first time, we recommend reading the main :class:`ActivationCache`
class first, including the examples, and then skimming the available methods. You can then refer
back to these docs depending on what you need to do.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from typing_extensions import Literal

import transformer_lens.utils as utils
from transformer_lens.utils import Slice, SliceInput


class ActivationCache:
    """Activation Cache.

    A wrapper that stores all important activations from a forward pass of the model, and provides a
    variety of helper functions to investigate them.

    The :class:`ActivationCache` is at the core of Transformer Lens. It is a wrapper that stores all
    important activations from a forward pass of the model, and provides a variety of helper
    functions to investigate them. The common way to access it is to run the model with
    :meth:`transformer_lens.HookedTransformer.run_with_cache`.

    Examples:

    When investigating a particular behaviour of a model, a very common first step is to try and
    understand which components of the model are most responsible for that behaviour. For example,
    if you're investigating the prompt "Why did the chicken cross the" -> " road", you might want to
    understand if there is a specific sublayer (mlp or multi-head attention) that is responsible for
    the model predicting "road". This kind of analysis commonly falls under the category of "logit
    attribution" or "direct logit attribution" (DLA).

    >>> from transformer_lens import HookedTransformer
    >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
    Loaded pretrained model tiny-stories-1M into HookedTransformer

    >>> _logits, cache = model.run_with_cache("Why did the chicken cross the")
    >>> residual_stream, labels = cache.decompose_resid(return_labels=True, mode="attn")
    >>> print(labels[0:3])
    ['embed', 'pos_embed', '0_attn_out']

    >>> answer = " road" # Note the proceeding space to match the model's tokenization
    >>> logit_attrs = cache.logit_attrs(residual_stream, answer)
    >>> print(logit_attrs.shape) # Attention layers
    torch.Size([10, 1, 7])

    >>> most_important_component_idx = torch.argmax(logit_attrs)
    >>> print(labels[most_important_component_idx])
    3_attn_out

    You can also dig in with more granularity, using :meth:`get_full_resid_decomposition` to get the
    residual stream by individual component (mlp neurons and individual attention heads). This
    creates a larger residual stack, but the approach of using :meth"`logit_attrs` remains the same.

    Equally you might want to find out if the model struggles to construct such excellent jokes
    until the very last layers, or if it is trivial and the first few layers are enough. This kind
    of analysis is called "logit lens", and you can find out more about how to do that with
    :meth:`ActivationCache.accumulated_resid`.

    Warning:

    :class:`ActivationCache` is designed to be used with
    :class:`transformer_lens.HookedTransformer`, and will not work with other models. It's also
    designed to be used with all activations of :class:`transformer_lens.HookedTransformer` being
    cached, and some internal methods will break without that.

    The biggest footgun and source of bugs in this code will be keeping track of indexes,
    dimensions, and the numbers of each. There are several kinds of activations:

    * Internal attn head vectors: q, k, v, z. Shape [batch, pos, head_index, d_head].
    * Internal attn pattern style results: pattern (post softmax), attn_scores (pre-softmax). Shape
      [batch, head_index, query_pos, key_pos].
    * Attn head results: result. Shape [batch, pos, head_index, d_model].
    * Internal MLP vectors: pre, post, mid (only used for solu_ln - the part between activation +
      layernorm). Shape [batch, pos, d_mlp].
    * Residual stream vectors: resid_pre, resid_mid, resid_post, attn_out, mlp_out, embed,
      pos_embed, normalized (output of each LN or LNPre). Shape [batch, pos, d_model].
    * LayerNorm Scale: scale. Shape [batch, pos, 1].

    Sometimes the batch dimension will be missing because we applied `remove_batch_dim` (used when
    batch_size=1), and as such all library functions *should* be robust to that.

    Type annotations are in the following form:

    * layers_covered is the number of layers queried in functions that stack the residual stream.
    * batch_and_pos_dims is the set of dimensions from batch and pos - by default this is ["batch",
      "pos"], but is only ["pos"] if we've removed the batch dimension and is [()] if we've removed
      batch dimension and are applying a pos slice which indexes a specific position.

    Args:
        cache_dict:
            A dictionary of cached activations from a model run.
        model:
            The model that the activations are from.
        has_batch_dim:
            Whether the activations have a batch dimension.
    """

    def __init__(self, cache_dict: Dict[str, torch.Tensor], model, has_batch_dim: bool = True):
        self.cache_dict = cache_dict
        self.model = model
        self.has_batch_dim = has_batch_dim
        self.has_embed = "hook_embed" in self.cache_dict
        self.has_pos_embed = "hook_pos_embed" in self.cache_dict

    def remove_batch_dim(self) -> ActivationCache:
        """Remove the Batch Dimension (if a single batch item).

        Returns:
            The ActivationCache with the batch dimension removed.
        """
        if self.has_batch_dim:
            for key in self.cache_dict:
                assert (
                    self.cache_dict[key].size(0) == 1
                ), f"Cannot remove batch dimension from cache with batch size > 1, \
                    for key {key} with shape {self.cache_dict[key].shape}"
                self.cache_dict[key] = self.cache_dict[key][0]
            self.has_batch_dim = False
        else:
            logging.warning("Tried removing batch dimension after already having removed it.")
        return self

    def __repr__(self) -> str:
        """Representation of the ActivationCache.

        Special method that returns a string representation of an object. It's normally used to give
        a string that can be used to recreate the object, but here we just return a string that
        describes the object.
        """
        return f"ActivationCache with keys {list(self.cache_dict.keys())}"

    def __getitem__(self, key) -> torch.Tensor:
        """Retrieve Cached Activations by Key or Shorthand.

        Enables direct access to cached activations via dictionary-style indexing using keys or
        shorthand naming conventions. It also supports tuples for advanced indexing, with the
        dimension order as (get_act_name, layer_index, layer_type).

        Args:
            key:
                The key or shorthand name for the activation to retrieve.

        Returns:
            The cached activation tensor corresponding to the given key.
        """
        if key in self.cache_dict:
            return self.cache_dict[key]
        elif type(key) == str:
            return self.cache_dict[utils.get_act_name(key)]
        else:
            if len(key) > 1 and key[1] is not None:
                if key[1] < 0:
                    # Supports negative indexing on the layer dimension
                    key = (key[0], self.model.cfg.n_layers + key[1], *key[2:])
            return self.cache_dict[utils.get_act_name(*key)]

    def __len__(self) -> int:
        """Length of the ActivationCache.

        Special method that returns the length of an object (in this case the number of different
        activations in the cache).
        """
        return len(self.cache_dict)

    def to(self, device: Union[str, torch.device], move_model=False) -> ActivationCache:
        """Move the Cache to a Device.

        Mostly useful for moving the cache to the CPU after model computation finishes to save GPU
        memory. Note however that operations will be much slower on the CPU. Note also that some
        methods will break unless the model is also moved to the same device, eg
        `compute_head_results`.

        Args:
            device:
                The device to move the cache to (e.g. `torch.device.cpu`).
            move_model:
                Whether to also move the model to the same device. @deprecated

        """
        # Move model is deprecated as we plan on de-coupling the classes
        if move_model is not None:
            warnings.warn(
                "The 'move_model' parameter is deprecated.",
                DeprecationWarning,
            )

        self.cache_dict = {key: value.to(device) for key, value in self.cache_dict.items()}

        if move_model:
            self.model.to(device)

        return self

    def toggle_autodiff(self, mode: bool = False):
        """Toggle Autodiff Globally.

        Applies `torch.set_grad_enabled(mode)` to the global state (not just TransformerLens).

        Warning:

        This is pretty dangerous, since autodiff is global state - this turns off torch's
        ability to take gradients completely and it's easy to get a bunch of errors if you don't
        realise what you're doing.

        But autodiff consumes a LOT of GPU memory (since every intermediate activation is cached
        until all downstream activations are deleted - this means that computing the loss and
        storing it in a list will keep every activation sticking around!). So often when you're
        analysing a model's activations, and don't need to do any training, autodiff is more trouble
        than its worth.

        If you don't want to mess with global state, using torch.inference_mode as a context manager
        or decorator achieves similar effects:

        >>> with torch.inference_mode():
        ...     y = torch.Tensor([1., 2, 3])
        >>> y.requires_grad
        False
        """
        logging.warning("Changed the global state, set autodiff to %s", mode)
        torch.set_grad_enabled(mode)

    def keys(self):
        """Keys of the ActivationCache.

        Examples:

            >>> from transformer_lens import HookedTransformer
            >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
            Loaded pretrained model tiny-stories-1M into HookedTransformer
            >>> _logits, cache = model.run_with_cache("Some prompt")
            >>> list(cache.keys())[0:3]
            ['hook_embed', 'hook_pos_embed', 'blocks.0.hook_resid_pre']

        Returns:
            List of all keys.
        """
        return self.cache_dict.keys()

    def values(self):
        """Values of the ActivationCache.

        Returns:
            List of all values.
        """
        return self.cache_dict.values()

    def items(self):
        """Items of the ActivationCache.

        Returns:
            List of all items ((key, value) tuples).
        """
        return self.cache_dict.items()

    def __iter__(self) -> Iterator[str]:
        """ActivationCache Iterator.

        Special method that returns an iterator over the ActivationCache. Allows looping over the
        cache.

        Examples:

            >>> from transformer_lens import HookedTransformer
            >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
            Loaded pretrained model tiny-stories-1M into HookedTransformer
            >>> _logits, cache = model.run_with_cache("Some prompt")
            >>> cache_interesting_names = []
            >>> for key in cache:
            ...     if not key.startswith("blocks.") or key.startswith("blocks.0"):
            ...         cache_interesting_names.append(key)
            >>> print(cache_interesting_names[0:3])
            ['hook_embed', 'hook_pos_embed', 'blocks.0.hook_resid_pre']

        Returns:
            Iterator over the cache.
        """
        return self.cache_dict.__iter__()

    def apply_slice_to_batch_dim(self, batch_slice: Union[Slice, SliceInput]) -> ActivationCache:
        """Apply a Slice to the Batch Dimension.

        Args:
            batch_slice:
                The slice to apply to the batch dimension.

        Returns:
            The ActivationCache with the batch dimension sliced.
        """
        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)
        batch_slice = cast(Slice, batch_slice)  # mypy can't seem to infer this
        assert (
            self.has_batch_dim or batch_slice.mode == "empty"
        ), "Cannot index into a cache without a batch dim"
        still_has_batch_dim = (batch_slice.mode != "int") and self.has_batch_dim
        new_cache_dict = {
            name: batch_slice.apply(param, dim=0) for name, param in self.cache_dict.items()
        }
        return ActivationCache(new_cache_dict, self.model, has_batch_dim=still_has_batch_dim)

    def accumulated_resid(
        self,
        layer: Optional[int] = None,
        incl_mid: bool = False,
        apply_ln: bool = False,
        pos_slice: Optional[Union[Slice, SliceInput]] = None,
        mlp_input: bool = False,
        return_labels: bool = False,
    ) -> Union[
        Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"], List[str]],
    ]:
        """Accumulated Residual Stream.

        Returns the accumulated residual stream at each layer/sub-layer. This is useful for `Logit
        Lens <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>`
        style analysis, where it can be thought of as what the model "believes" at each point in the
        residual stream.

        To project this into the vocabulary space, remember that there is a final layer norm in most
        decoder-only transformers. Therefore, you need to first apply the final layer norm (which
        can be done with `apply_ln`), and then multiply by the unembedding matrix (:math:`W_U`).

        If you instead want to look at contributions to the residual stream from each component
        (e.g. for direct logit attribution), see :meth:`decompose_resid` instead, or
        :meth:`get_full_resid_decomposition` if you want contributions broken down further into each
        MLP neuron.

        Examples:

        Logit Lens analysis can be done as follows:

        >>> from transformer_lens import HookedTransformer
        >>> from einops import einsum
        >>> import torch
        >>> import pandas as pd

        >>> model = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
        Loaded pretrained model tiny-stories-1M into HookedTransformer

        >>> prompt = "Why did the chicken cross the"
        >>> answer = " road"
        >>> logits, cache = model.run_with_cache("Why did the chicken cross the")
        >>> answer_token = model.to_single_token(answer)
        >>> print(answer_token)
        2975

        >>> accum_resid, labels = cache.accumulated_resid(return_labels=True, apply_ln=True)
        >>> last_token_accum = accum_resid[:, 0, -1, :]  # layer, batch, pos, d_model
        >>> print(last_token_accum.shape)  # layer, d_model
        torch.Size([9, 64])

        >>> W_U = model.W_U
        >>> print(W_U.shape)
        torch.Size([64, 50257])

        >>> layers_unembedded = einsum(
        ...         last_token_accum,
        ...         W_U,
        ...         "layer d_model, d_model d_vocab -> layer d_vocab"
        ...     )
        >>> print(layers_unembedded.shape)
        torch.Size([9, 50257])

        >>> # Get the rank of the correct answer by layer
        >>> sorted_indices = torch.argsort(layers_unembedded, dim=1, descending=True)
        >>> rank_answer = (sorted_indices == 2975).nonzero(as_tuple=True)[1]
        >>> print(pd.Series(rank_answer, index=labels))
        0_pre         4442
        1_pre          382
        2_pre          982
        3_pre         1160
        4_pre          408
        5_pre          145
        6_pre           78
        7_pre          387
        final_post       6
        dtype: int64

        Args:
            layer:
                The layer to take components up to - by default includes resid_pre for that layer
                and excludes resid_mid and resid_post for that layer. If set as `n_layers`, `-1` or
                `None` it will return all residual streams, including the final one (i.e.
                immediately pre logits). The indices are taken such that this gives the accumulated
                streams up to the input to layer l.
            incl_mid:
                Whether to return `resid_mid` for all previous layers.
            apply_ln:
                Whether to apply LayerNorm to the stack.
            pos_slice:
                A slice object to apply to the pos dimension. Defaults to None, do nothing.
            mlp_input:
                Whether to include resid_mid for the current layer. This essentially gives the MLP
                input rather than the attention input.
            return_labels:
                Whether to return a list of labels for the residual stream components. Useful for
                labelling graphs.

        Returns:
            A tensor of the accumulated residual streams. If `return_labels` is True, also returns a
            list of labels for the components (as a tuple in the form `(components, labels)`).
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers
        assert isinstance(layer, int)
        labels = []
        components_list = []
        for l in range(layer + 1):
            if l == self.model.cfg.n_layers:
                components_list.append(self[("resid_post", self.model.cfg.n_layers - 1)])
                labels.append("final_post")
                continue
            components_list.append(self[("resid_pre", l)])
            labels.append(f"{l}_pre")
            if (incl_mid and l < layer) or (mlp_input and l == layer):
                components_list.append(self[("resid_mid", l)])
                labels.append(f"{l}_mid")
        components_list = [pos_slice.apply(c, dim=-2) for c in components_list]
        components = torch.stack(components_list, dim=0)
        if apply_ln:
            components = self.apply_ln_to_stack(
                components, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )
        if return_labels:
            return components, labels
        else:
            return components

    def logit_attrs(
        self,
        residual_stack: Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        tokens: Union[
            str,
            int,
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "batch"],
            Int[torch.Tensor, "batch position"],
        ],
        incorrect_tokens: Optional[
            Union[
                str,
                int,
                Int[torch.Tensor, ""],
                Int[torch.Tensor, "batch"],
                Int[torch.Tensor, "batch position"],
            ]
        ] = None,
        pos_slice: Union[Slice, SliceInput] = None,
        batch_slice: Union[Slice, SliceInput] = None,
        has_batch_dim: bool = True,
    ) -> Float[torch.Tensor, "num_components *batch_and_pos_dims_out"]:
        """Logit Attributions.

        Takes a residual stack (typically the residual stream decomposed by components), and
        calculates how much each item in the stack "contributes" to specific tokens.

        It does this by:
            1. Getting the residual directions of the tokens (i.e. reversing the unembed)
            2. Taking the dot product of each item in the residual stack, with the token residual
               directions.

        Note that if incorrect tokens are provided, it instead takes the difference between the
        correct and incorrect tokens (to calculate the residual directions). This is useful as
        sometimes we want to know e.g. which components are most responsible for selecting the
        correct token rather than an incorrect one. For example in the `Interpretability in the Wild
        paper <https://arxiv.org/abs/2211.00593>` prompts such as "John and Mary went to the shops,
        John gave a bag to" were investigated, and it was therefore useful to calculate attribution
        for the :math:`\\text{Mary} - \\text{John}` residual direction.

        Warning:

        Choosing the correct `tokens` and `incorrect_tokens` is both important and difficult. When
        investigating specific components it's also useful to look at it's impact on all tokens
        (i.e. :math:`\\text{final_ln}(\\text{residual_stack_item}) W_U`).

        Args:
            residual_stack:
                Stack of components of residual stream to get logit attributions for.
            tokens:
                Tokens to compute logit attributions on.
            incorrect_tokens:
                If provided, compute attributions on logit difference between tokens and
                incorrect_tokens. Must have the same shape as tokens.
            pos_slice:
                The slice to apply layer norm scaling on. Defaults to None, do nothing.
            batch_slice:
                The slice to take on the batch dimension during layer norm scaling. Defaults to
                None, do nothing.
            has_batch_dim:
                Whether residual_stack has a batch dimension. Defaults to True.

        Returns:
            A tensor of the logit attributions or logit difference attributions if incorrect_tokens
            was provided.
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)

        if isinstance(tokens, str):
            tokens = torch.as_tensor(self.model.to_single_token(tokens))

        elif isinstance(tokens, int):
            tokens = torch.as_tensor(tokens)

        logit_directions = self.model.tokens_to_residual_directions(tokens)

        if incorrect_tokens is not None:
            if isinstance(incorrect_tokens, str):
                incorrect_tokens = torch.as_tensor(self.model.to_single_token(incorrect_tokens))

            elif isinstance(incorrect_tokens, int):
                incorrect_tokens = torch.as_tensor(incorrect_tokens)

            if tokens.shape != incorrect_tokens.shape:
                raise ValueError(
                    f"tokens and incorrect_tokens must have the same shape! \
                        (tokens.shape={tokens.shape}, \
                        incorrect_tokens.shape={incorrect_tokens.shape})"
                )

            # If incorrect_tokens was provided, take the logit difference
            logit_directions = logit_directions - self.model.tokens_to_residual_directions(
                incorrect_tokens
            )

        scaled_residual_stack = self.apply_ln_to_stack(
            residual_stack,
            layer=-1,
            pos_slice=pos_slice,
            batch_slice=batch_slice,
            has_batch_dim=has_batch_dim,
        )

        # Element-wise multiplication and sum over the d_model dimension
        logit_attrs = (scaled_residual_stack * logit_directions).sum(dim=-1)
        return logit_attrs

    def decompose_resid(
        self,
        layer: Optional[int] = None,
        mlp_input: bool = False,
        mode: Literal["all", "mlp", "attn"] = "all",
        apply_ln: bool = False,
        pos_slice: Union[Slice, SliceInput] = None,
        incl_embeds: bool = True,
        return_labels: bool = False,
    ) -> Union[
        Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"], List[str]],
    ]:
        """Decompose the Residual Stream.

        Decomposes the residual stream input to layer L into a stack of the output of previous
        layers. The sum of these is the input to layer L (plus embedding and pos embedding). This is
        useful for attributing model behaviour to different components of the residual stream

        Args:
            layer:
                The layer to take components up to - by default includes
                resid_pre for that layer and excludes resid_mid and resid_post for that layer.
                layer==n_layers means to return all layer outputs incl in the final layer, layer==0
                means just embed and pos_embed. The indices are taken such that this gives the
                accumulated streams up to the input to layer l
            mlp_input:
                Whether to include attn_out for the current
                layer - essentially decomposing the residual stream that's input to the MLP input
                rather than the Attn input.
            mode:
                Values are "all", "mlp" or "attn". "all" returns all
                components, "mlp" returns only the MLP components, and "attn" returns only the
                attention components. Defaults to "all".
            apply_ln:
                Whether to apply LayerNorm to the stack.
            pos_slice:
                A slice object to apply to the pos dimension.
                Defaults to None, do nothing.
            incl_embeds:
                Whether to include embed & pos_embed
            return_labels:
                Whether to return a list of labels for the residual stream components.
                Useful for labelling graphs.

        Returns:
            A tensor of the accumulated residual streams. If `return_labels` is True, also returns
            a list of labels for the components (as a tuple in the form `(components, labels)`).
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        pos_slice = cast(Slice, pos_slice)  # mypy can't seem to infer this
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers
        assert isinstance(layer, int)

        incl_attn = mode != "mlp"
        incl_mlp = mode != "attn" and not self.model.cfg.attn_only
        components_list = []
        labels = []
        if incl_embeds:
            if self.has_embed:
                components_list = [self["hook_embed"]]
                labels.append("embed")
            if self.has_pos_embed:
                components_list.append(self["hook_pos_embed"])
                labels.append("pos_embed")

        for l in range(layer):
            if incl_attn:
                components_list.append(self[("attn_out", l)])
                labels.append(f"{l}_attn_out")
            if incl_mlp:
                components_list.append(self[("mlp_out", l)])
                labels.append(f"{l}_mlp_out")
        if mlp_input and incl_attn:
            components_list.append(self[("attn_out", layer)])
            labels.append(f"{layer}_attn_out")
        components_list = [pos_slice.apply(c, dim=-2) for c in components_list]
        components = torch.stack(components_list, dim=0)
        if apply_ln:
            components = self.apply_ln_to_stack(
                components, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )
        if return_labels:
            return components, labels
        else:
            return components

    def compute_head_results(
        self,
    ):
        """Compute Head Results.

        Computes and caches the results for each attention head, ie the amount contributed to the
        residual stream from that head. attn_out for a layer is the sum of head results plus b_O.
        Intended use is to enable use_attn_results when running and caching the model, but this can
        be useful if you forget.
        """
        if "blocks.0.attn.hook_result" in self.cache_dict:
            logging.warning("Tried to compute head results when they were already cached")
            return
        for layer in range(self.model.cfg.n_layers):
            # Note that we haven't enabled set item on this object so we need to edit the underlying
            # cache_dict directly.

            # Add singleton dimension to match W_O's shape for broadcasting
            z = einops.rearrange(
                self[("z", layer, "attn")],
                "... head_index d_head -> ... head_index d_head 1",
            )

            # Element-wise multiplication of z and W_O (with shape [head_index, d_head, d_model])
            result = z * self.model.blocks[layer].attn.W_O

            # Sum over d_head to get the contribution of each head to the residual stream
            self.cache_dict[f"blocks.{layer}.attn.hook_result"] = result.sum(dim=-2)

    def stack_head_results(
        self,
        layer: int = -1,
        return_labels: bool = False,
        incl_remainder: bool = False,
        pos_slice: Union[Slice, SliceInput] = None,
        apply_ln: bool = False,
    ) -> Union[
        Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"], List[str]],
    ]:
        """Stack Head Results.

        Returns a stack of all head results (ie residual stream contribution) up to layer L. A good
        way to decompose the outputs of attention layers into attribution by specific heads. Note
        that the num_components axis has length layer x n_heads ((layer head_index) in einops
        notation).

        Args:
            layer:
                Layer index - heads at all layers strictly before this are included. layer must be
                in [1, n_layers-1], or any of (n_layers, -1, None), which all mean the final layer.
            return_labels:
                Whether to also return a list of labels of the form "L0H0" for the heads.
            incl_remainder:
                Whether to return a final term which is "the rest of the residual stream".
            pos_slice:
                A slice object to apply to the pos dimension. Defaults to None, do nothing.
            apply_ln:
                Whether to apply LayerNorm to the stack.
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        pos_slice = cast(Slice, pos_slice)  # mypy can't seem to infer this
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        if "blocks.0.attn.hook_result" not in self.cache_dict:
            print(
                "Tried to stack head results when they weren't cached. Computing head results now"
            )
            self.compute_head_results()

        components: Any = []
        labels = []
        for l in range(layer):
            # Note that this has shape batch x pos x head_index x d_model
            components.append(pos_slice.apply(self[("result", l, "attn")], dim=-3))
            labels.extend([f"L{l}H{h}" for h in range(self.model.cfg.n_heads)])
        if components:
            components = torch.cat(components, dim=-2)
            components = einops.rearrange(
                components,
                "... concat_head_index d_model -> concat_head_index ... d_model",
            )
            if incl_remainder:
                remainder = pos_slice.apply(
                    self[("resid_post", layer - 1)], dim=-2
                ) - components.sum(dim=0)
                components = torch.cat([components, remainder[None]], dim=0)
                labels.append("remainder")
        elif incl_remainder:
            # There are no components, so the remainder is the entire thing.
            components = torch.cat(
                [pos_slice.apply(self[("resid_post", layer - 1)], dim=-2)[None]], dim=0
            )
            labels.append("remainder")
        else:
            # If this is called with layer 0, we return an empty tensor of the right shape to be
            # stacked correctly. This uses the shape of hook_embed, which is pretty janky since it
            # assumes embed is in the cache. But it's hard to explicitly code the shape, since it
            # depends on the pos slice, whether we have a batch dim, etc. And it's pretty messy!
            components = torch.zeros(
                0,
                *pos_slice.apply(self["hook_embed"], dim=-2).shape,
                device=self.model.cfg.device,
            )

        if apply_ln:
            components = self.apply_ln_to_stack(components, layer, pos_slice=pos_slice)

        if return_labels:
            return components, labels
        else:
            return components

    def stack_activation(
        self,
        activation_name: str,
        layer: int = -1,
        sublayer_type: Optional[str] = None,
    ) -> Float[torch.Tensor, "layers_covered ..."]:
        """Stack Activations.

        Flexible way to stack activations with a given name.

        Args:
            activation_name:
                The name of the activation to be stacked
            layer:
                'Layer index - heads' at all layers strictly before this are included. layer must be
                in [1, n_layers-1], or any of (n_layers, -1, None), which all mean the final layer.
            sublayer_type:
                The sub layer type of the activation, passed to utils.get_act_name. Can normally be
                inferred.
            incl_remainder:
                Whether to return a final term which is "the rest of the residual stream".
        """
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        components = []
        for l in range(layer):
            components.append(self[(activation_name, l, sublayer_type)])

        return torch.stack(components, dim=0)

    def get_neuron_results(
        self,
        layer: int,
        neuron_slice: Union[Slice, SliceInput] = None,
        pos_slice: Union[Slice, SliceInput] = None,
    ) -> Float[torch.Tensor, "*batch_and_pos_dims num_neurons d_model"]:
        """Get Neuron Results.

        Get the results of for neurons in a specific layer (i.e, how much each neuron contributes to
        the residual stream). Does it for the subset of neurons specified by neuron_slice, defaults
        to all of them. Does *not* cache these because it's expensive in space and cheap to compute.

        Args:
            layer:
                Layer index.
            neuron_slice:
                Slice of the neuron.
            pos_slice:
                Slice of the positions.

        Returns:
            Tensor of the results.
        """
        if not isinstance(neuron_slice, Slice):
            neuron_slice = Slice(neuron_slice)
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        neuron_acts = self[("post", layer, "mlp")]
        W_out = self.model.blocks[layer].mlp.W_out
        if pos_slice is not None:
            # Note - order is important, as Slice.apply *may* collapse a dimension, so this ensures
            # that position dimension is -2 when we apply position slice
            neuron_acts = pos_slice.apply(neuron_acts, dim=-2)
        if neuron_slice is not None:
            neuron_acts = neuron_slice.apply(neuron_acts, dim=-1)
            W_out = neuron_slice.apply(W_out, dim=0)
        return neuron_acts[..., None] * W_out

    def stack_neuron_results(
        self,
        layer: int,
        pos_slice: Union[Slice, SliceInput] = None,
        neuron_slice: Union[Slice, SliceInput] = None,
        return_labels: bool = False,
        incl_remainder: bool = False,
        apply_ln: bool = False,
    ) -> Union[
        Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"], List[str]],
    ]:
        """Stack Neuron Results

        Returns a stack of all neuron results (ie residual stream contribution) up to layer L - ie
        the amount each individual neuron contributes to the residual stream. Also returns a list of
        labels of the form "L0N0" for the neurons. A good way to decompose the outputs of MLP layers
        into attribution by specific neurons.

        Note that doing this for all neurons is SUPER expensive on GPU memory and only works for
        small models or short inputs.

        Args:
            layer:
                Layer index - heads at all layers strictly before this are included. layer must be
                in [1, n_layers]
            pos_slice:
                Slice of the positions.
            neuron_slice:
                Slice of the neurons.
            return_labels:
                Whether to also return a list of labels of the form "L0H0" for the heads.
            incl_remainder:
                Whether to return a final term which is "the rest of the residual stream".
            apply_ln:
                Whether to apply LayerNorm to the stack.
        """

        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        components: Any = []  # TODO: fix typing properly
        labels = []

        if not isinstance(neuron_slice, Slice):
            neuron_slice = Slice(neuron_slice)
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        neuron_labels: torch.Tensor | np.ndarray = neuron_slice.apply(
            torch.arange(self.model.cfg.d_mlp), dim=0
        )
        if type(neuron_labels) == int:
            neuron_labels = np.array([neuron_labels])
        for l in range(layer):
            # Note that this has shape batch x pos x head_index x d_model
            components.append(
                self.get_neuron_results(l, pos_slice=pos_slice, neuron_slice=neuron_slice)
            )
            labels.extend([f"L{l}N{h}" for h in neuron_labels])
        if components:
            components = torch.cat(components, dim=-2)
            components = einops.rearrange(
                components,
                "... concat_neuron_index d_model -> concat_neuron_index ... d_model",
            )

            if incl_remainder:
                remainder = pos_slice.apply(
                    self[("resid_post", layer - 1)], dim=-2
                ) - components.sum(dim=0)
                components = torch.cat([components, remainder[None]], dim=0)
                labels.append("remainder")
        elif incl_remainder:
            components = torch.cat(
                [pos_slice.apply(self[("resid_post", layer - 1)], dim=-2)[None]], dim=0
            )
            labels.append("remainder")
        else:
            # Returning empty, give it the right shape to stack properly
            components = torch.zeros(
                0,
                *pos_slice.apply(self["hook_embed"], dim=-2).shape,
                device=self.model.cfg.device,
            )

        if apply_ln:
            components = self.apply_ln_to_stack(components, layer, pos_slice=pos_slice)

        if return_labels:
            return components, labels
        else:
            return components

    def apply_ln_to_stack(
        self,
        residual_stack: Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        layer: Optional[int] = None,
        mlp_input: bool = False,
        pos_slice: Union[Slice, SliceInput] = None,
        batch_slice: Union[Slice, SliceInput] = None,
        has_batch_dim: bool = True,
    ) -> Float[torch.Tensor, "num_components *batch_and_pos_dims_out d_model"]:
        """Apply Layer Norm to a Stack.

        Takes a stack of components of the residual stream (eg outputs of decompose_resid or
        accumulated_resid), treats them as the input to a specific layer, and applies the layer norm
        scaling of that layer to them, using the cached scale factors - simulating what that
        component of the residual stream contributes to that layer's input.

        The layernorm scale is global across the entire residual stream for each layer, batch
        element and position, which is why we need to use the cached scale factors rather than just
        applying a new LayerNorm.

        If the model does not use LayerNorm or RMSNorm, it returns the residual stack unchanged.

        Args:
            residual_stack:
                A tensor, whose final dimension is d_model. The other trailing dimensions are
                assumed to be the same as the stored hook_scale - which may or may not include batch
                or position dimensions.
            layer:
                The layer we're taking the input to. In [0, n_layers], n_layers means the unembed.
                None maps to the n_layers case, ie the unembed.
            mlp_input:
                Whether the input is to the MLP or attn (ie ln2 vs ln1). Defaults to False, ie ln1.
                If layer==n_layers, must be False, and we use ln_final
            pos_slice:
                The slice to take of positions, if residual_stack is not over the full context, None
                means do nothing. It is assumed that pos_slice has already been applied to
                residual_stack, and this is only applied to the scale. See utils.Slice for details.
                Defaults to None, do nothing.
            batch_slice:
                The slice to take on the batch dimension. Defaults to None, do nothing.
            has_batch_dim:
                Whether residual_stack has a batch dimension.

        """
        if self.model.cfg.normalization_type not in ["LN", "LNPre", "RMS", "RMSPre"]:
            # The model does not use LayerNorm, so we don't need to do anything.
            return residual_stack
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)

        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        if has_batch_dim:
            # Apply batch slice to the stack
            residual_stack = batch_slice.apply(residual_stack, dim=1)

        # Center the stack onlny if the model uses LayerNorm
        if self.model.cfg.normalization_type in ["LN", "LNPre"]:
            residual_stack = residual_stack - residual_stack.mean(dim=-1, keepdim=True)

        if layer == self.model.cfg.n_layers or layer is None:
            scale = self["ln_final.hook_scale"]
        else:
            hook_name = f"blocks.{layer}.ln{2 if mlp_input else 1}.hook_scale"
            scale = self[hook_name]

        # The shape of scale is [batch, position, 1] or [position, 1] - final dimension is a dummy
        # thing to get broadcoasting to work nicely.
        scale = pos_slice.apply(scale, dim=-2)

        if self.has_batch_dim:
            # Apply batch slice to the scale
            scale = batch_slice.apply(scale)

        return residual_stack / scale

    def get_full_resid_decomposition(
        self,
        layer: Optional[int] = None,
        mlp_input: bool = False,
        expand_neurons: bool = True,
        apply_ln: bool = False,
        pos_slice: Union[Slice, SliceInput] = None,
        return_labels: bool = False,
    ) -> Union[
        Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"], List[str]],
    ]:
        """Get the full Residual Decomposition.

        Returns the full decomposition of the residual stream into embed, pos_embed, each head
        result, each neuron result, and the accumulated biases. We break down the residual stream
        that is input into some layer.

        Args:
            layer:
                The layer we're inputting into. layer is in [0, n_layers], if layer==n_layers (or
                None) we're inputting into the unembed (the entire stream), if layer==0 then it's
                just embed and pos_embed
            mlp_input:
                Are we inputting to the MLP in that layer or the attn? Must be False for final
                layer, since that's the unembed.
            expand_neurons:
                Whether to expand the MLP outputs to give every neuron's result or just return the
                MLP layer outputs.
            apply_ln:
                Whether to apply LayerNorm to the stack.
            pos_slice:
                Slice of the positions to take.
            return_labels:
                Whether to return the labels.
        """
        if layer is None or layer == -1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers
        assert layer is not None  # keep mypy happy

        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        head_stack, head_labels = self.stack_head_results(
            layer + (1 if mlp_input else 0), pos_slice=pos_slice, return_labels=True
        )
        labels = head_labels
        components = [head_stack]
        if not self.model.cfg.attn_only and layer > 0:
            if expand_neurons:
                neuron_stack, neuron_labels = self.stack_neuron_results(
                    layer, pos_slice=pos_slice, return_labels=True
                )
                labels.extend(neuron_labels)
                components.append(neuron_stack)
            else:
                # Get the stack of just the MLP outputs
                # mlp_input included for completeness, but it doesn't actually matter, since it's
                # just for MLP outputs
                mlp_stack, mlp_labels = self.decompose_resid(
                    layer,
                    mlp_input=mlp_input,
                    pos_slice=pos_slice,
                    incl_embeds=False,
                    mode="mlp",
                    return_labels=True,
                )
                labels.extend(mlp_labels)
                components.append(mlp_stack)

        if self.has_embed:
            labels.append("embed")
            components.append(pos_slice.apply(self["embed"], -2)[None])
        if self.has_pos_embed:
            labels.append("pos_embed")
            components.append(pos_slice.apply(self["pos_embed"], -2)[None])
        # If we didn't expand the neurons, the MLP biases are already included in the MLP outputs.
        bias = self.model.accumulated_bias(layer, mlp_input, include_mlp_biases=expand_neurons)
        bias = bias.expand((1,) + head_stack.shape[1:])
        labels.append("bias")
        components.append(bias)
        residual_stack = torch.cat(components, dim=0)
        if apply_ln:
            residual_stack = self.apply_ln_to_stack(
                residual_stack, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )

        if return_labels:
            return residual_stack, labels
        else:
            return residual_stack
