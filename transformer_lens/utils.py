"""Utils.

This module contains varied utility functions used throughout the library.
"""

from __future__ import annotations

import inspect
import io
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from jaxtyping import Float, Int
from rich import print as rprint
from transformers import AutoTokenizer

from transformer_lens.FactoredMatrix import FactoredMatrix

CACHE_DIR = transformers.TRANSFORMERS_CACHE
USE_DEFAULT_VALUE = None


def select_compatible_kwargs(kwargs_dict: Dict[str, Any], callable: Callable) -> Dict[str, Any]:
    """Return a dict with the elements kwargs_dict that are parameters of callable"""
    return {k: v for k, v in kwargs_dict.items() if k in inspect.getfullargspec(callable).args}


def download_file_from_hf(
    repo_name,
    file_name,
    subfolder=".",
    cache_dir=CACHE_DIR,
    force_is_torch=False,
    **kwargs,
):
    """
    Helper function to download files from the HuggingFace Hub, from subfolder/file_name in repo_name, saving locally to cache_dir and returning the loaded file (if a json or Torch object) and the file path otherwise.

    If it's a Torch file without the ".pth" extension, set force_is_torch=True to load it as a Torch object.
    """
    file_path = hf_hub_download(
        repo_id=repo_name,
        filename=file_name,
        subfolder=subfolder,
        cache_dir=cache_dir,
        **select_compatible_kwargs(kwargs, hf_hub_download),
    )

    if file_path.endswith(".pth") or force_is_torch:
        return torch.load(file_path, map_location="cpu", weights_only=False)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path


def upload_model_to_hf(model: "HookedTransformer", repo_name: str, commit_message: str = None):
    """
    Upload a model to the Hugging Face Hub.
    """
    api = HfApi()
    config_buffer = io.BytesIO()
    config_buffer.write(model.cfg.to_json(indent=2).encode("utf-8"))
    config_buffer.seek(0)
    add_config = CommitOperationAdd(path_or_fileobj=config_buffer, path_in_repo="tl_config.json")

    with tempfile.TemporaryFile() as f:
        torch.save(model.state_dict(), f)
        f.seek(0)
        add_model = CommitOperationAdd(path_or_fileobj=f, path_in_repo="state_dict.pth")
        api.create_commit(
            repo_id=repo_name,
            operations=[add_config, add_model],
            commit_message=commit_message,
        )


def clear_huggingface_cache():
    """
    Deletes the Hugging Face cache directory and all its contents.

    This function deletes the Hugging Face cache directory, which is used to store downloaded models and their associated files. Deleting the cache directory will remove all the downloaded models and their files, so you will need to download them again if you want to use them in your code.

    Parameters:
    None

    Returns:
    None
    """
    print("Deleting Hugging Face cache directory and all its contents.")
    shutil.rmtree(CACHE_DIR)


def print_gpu_mem(step_name=""):
    print(f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30, 2)} GiB allocated on GPU.")


def get_corner(tensor, n=3):
    # Prints the top left corner of the tensor
    if isinstance(tensor, torch.Tensor):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))]
    elif isinstance(tensor, FactoredMatrix):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))].AB


def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def lm_cross_entropy_loss(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    tokens: Int[torch.Tensor, "batch pos"],
    attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        attention_mask (torch.Tensor[int64], optional): Attention mask. Shape [batch, pos]. Used to
            mask out padding tokens. Defaults to None.
        per_token (bool, optional): Whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs). Note that the returned array has shape [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final logit). Defaults to False.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(dim=-1, index=tokens[..., 1:, None])[..., 0]

    if attention_mask is not None:
        # Ignore token positions which are masked out or where the next token is masked out
        # (generally padding tokens)
        next_token_mask = torch.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
        predicted_log_probs *= next_token_mask
        n_tokens = next_token_mask.sum().item()
    else:
        n_tokens = predicted_log_probs.numel()

    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.sum() / n_tokens


def lm_accuracy(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    tokens: Int[torch.Tensor, "batch pos"],
    per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """Cross-Entropy Accuracy for Language Modelling. We measure the accuracy on the logits for predicting the NEXT token.

    If per_token is True, returns the boolean for top 1 accuracy for each token in the batch. Note that this has size [batch, seq_len-1], as we cannot predict the first token.
    """
    top_prediction = logits.argmax(dim=-1)
    correct_matches = top_prediction[:, :-1] == tokens[:, 1:]
    if per_token:
        return correct_matches
    else:
        return correct_matches.sum() / correct_matches.numel()


def gelu_new(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


def gelu_fast(
    input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


def solu(input: Float[torch.Tensor, "batch pos d_mlp"]) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.

    LayerNorm implemented by the MLP class.
    """
    return input * F.softmax(input, dim=-1)


ACTIVATION_FN_DICT = {
    "solu": solu,
    "solu_ln": solu,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda tensor: F.gelu(tensor, approximate="tanh"),
}


def calc_fan_in_and_fan_out(tensor):
    """
    Calculate the fan in and fan out of a tensor. We define it ourselves because Torch uses a
    different convention for weights (e.g. for an MLP they use d_out x d_in, and we use d_in x
    d_out, for attention they do (n_head d_head) x d_model, we do n_head x d_model x d_head).
    """
    shape = tensor.shape

    if len(shape) == 0:
        raise ValueError("Fan in and fan out can not be computed for scalars.")
    elif len(shape) == 1:
        fan_in = 1
        fan_out = shape[0]
    elif len(shape) == 2:  # Linear transform
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 3:  # Attention head weight, has shape n_head x d_model x d_head
        fan_in = shape[1]
        fan_out = shape[0] * shape[2]
    else:
        raise ValueError(f"Fan in and fan out can not be computed for shape {shape} tensors.")

    return fan_in, fan_out


def init_xavier_uniform_(param, gain=1.0):
    """
    Initializes the input tensor using the Xavier initialization method.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    max = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return nn.init.uniform_(param, -max, max)


def init_xavier_normal_(param, gain=1.0):
    """
    Initializes the input tensor using the Xavier initialization method.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return nn.init.normal_(param, mean=0.0, std=std)


def init_kaiming_uniform_(param, a=0, nonlinearity="relu", gain=1.0, mode="fan_in"):
    """
    Initializes the input tensor using the Kaiming initialization method.

    Starting from a std 1 uniform distribution, we scale the weights by c / sqrt(fan_in), where c =
    sqrt(2) if the params were immediately preceded by a relu and 1 for everything else.

    As with torch, `a` is a hyperparameter for `nonlinearity`, if it takes one.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    fan = fan_in if mode == "fan_in" else fan_out
    gain *= nn.init.calculate_gain(nonlinearity, a)
    max = gain * np.sqrt(3.0 / fan)
    return nn.init.uniform_(param, -max, max)


def init_kaiming_normal_(param, a=0, nonlinearity="relu", gain=1.0, mode="fan_in"):
    """
    Initializes the input tensor using the Kaiming initialization method.

    Starting from a std 1 normal distribution, we scale the weights by c / sqrt(fan_in), where c =
    sqrt(2) if the params were immediately preceded by a relu and 1 for everything else.

    As with torch, `a` is a hyperparameter for `nonlinearity`, if it takes one.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    fan = fan_in if mode == "fan_in" else fan_out
    gain *= nn.init.calculate_gain(nonlinearity, a)
    std = gain * np.sqrt(1.0 / fan)
    return nn.init.normal_(param, mean=0.0, std=std)


def keep_single_column(dataset: Dataset, col_name: str):
    """
    Acts on a HuggingFace dataset to delete all columns apart from a single column name - useful when we want to tokenize and mix together different strings
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
) -> Dataset:
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end.

    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"
    """
    dataset = keep_single_column(dataset, column_name)
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.eos_token.join(text)

        # Handle the case when full_text is empty
        if not full_text.strip():
            return {"tokens": np.array([], dtype=np.int64)}

        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()
        # Drop padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)

        # Handle cases where num_tokens is less than seq_len
        if num_tokens < seq_len:
            num_batches = 1
            # Pad tokens if necessary
            tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                padding_length = seq_len - len(tokens)
                padding = np.full(padding_length, tokenizer.pad_token_id)
                tokens = np.concatenate([tokens, padding], axis=0)
        else:
            num_batches = num_tokens // seq_len
            # Drop the final tokens if not enough to make a full sequence
            tokens = tokens[: seq_len * num_batches]

        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset


def sample_logits(
    final_logits: Float[torch.Tensor, "batch d_vocab"],
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    freq_penalty: float = 0.0,
    tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
) -> Int[torch.Tensor, "batch"]:
    """
    Sample from the logits, in order to generate text

    final_logits has shape [batch, vocab_size]
    We divide the logits by temperature before softmaxing and sampling - high temperature = more uniform, low = more argmaxy. Temp = 0.0 is greedy sampling
    We apply top_k and top_p filtering to the logits, to encourage diversity. top_k = 10 means we only sample from the 10 most likely tokens. top_p = 0.9 means we only sample from the top 90% of tokens, and then renormalise the distribution. top_k and top_p are mutually exclusive. By default we apply neither and just sample from the full distribution.

    Frequency penalty is a penalty on the probability of a token, proportional to the number of times it has been generated so far. This encourages the model to generate new tokens, rather than repeating itself. It is a hyperparameter, and should be tuned. It is applied to the logits before sampling. If this is non-zero it is required to input the input_tokens

    #! TODO: Finish testing all the edge cases here. Useful testing code:
    logits = torch.randn(4)
    print(logits)
    np.unique(np.array([sample_logits(logits, top_k=2).item() for i in range(1000)]), return_counts=True)
    """
    if temperature == 0.0:
        # Greedy sampling
        return final_logits.argmax(dim=-1)
    else:
        # Sample from the distribution

        final_logits = final_logits / temperature
        if freq_penalty > 0:
            assert tokens is not None, "Must provide input_tokens if applying a frequency penalty"
            for batch_index in range(final_logits.shape[0]):
                # torch.bincount returns a tensor of length d_vocab, with the number of occurences of each token in the tokens.
                final_logits[batch_index] = final_logits[
                    batch_index
                ] - freq_penalty * torch.bincount(
                    tokens[batch_index], minlength=final_logits.shape[-1]
                )
        if top_k is not None:
            assert top_k > 0, "top_k has to be greater than 0"
            top_logits, top_idx = final_logits.topk(top_k, dim=-1)
            indices_to_remove = final_logits < top_logits[..., -1].unsqueeze(-1)
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))
        elif top_p is not None:
            assert 1.0 >= top_p > 0.0, "top_p has to be in (0, 1]"
            sorted_logits, sorted_indices = torch.sort(final_logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # We round up - we want prob >= top_p not <top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))

        final_logits = final_logits.to(torch.float32)
        return torch.distributions.categorical.Categorical(logits=final_logits).sample()


# Type alias
SliceInput = Optional[
    Union[
        int,
        Tuple[int,],
        Tuple[int, int],
        Tuple[int, int, int],
        List[int],
        torch.Tensor,
        np.ndarray,
    ]
]
"""An object that represents a slice input. It can be a tuple of integers or a slice object.

An optional type alias for a slice input used in the `ActivationCache` module.

A `SliceInput` can be one of the following types:
    - `int`: an integer representing a single position
    - `Tuple[int, int]`: a tuple of two integers representing a range of positions
    - `Tuple[int, int, int]`: a tuple of three integers representing a range of positions with a step size
    - `List[int]`: a list of integers representing multiple positions
    - `torch.Tensor`: a tensor containing a boolean mask or a list of indices to be selected from the input tensor.

`SliceInput` is used in the `apply_ln_to_stack` method in the `ActivationCache` module.
"""


class Slice:
    """An object that represents a slice input. It can be a tuple of integers or a slice object.

    We use a custom slice syntax because Python/Torch's don't let us reduce the number of dimensions:

    Note that slicing with input_slice=None means do nothing, NOT add an extra dimension (use unsqueeze for that)

    There are several modes:
    int - just index with that integer (decreases number of dimensions)
    slice - Input is a tuple converted to a slice ((k,) means :k, (k, m) means m:k, (k, m, n) means m:k:n)
    array - Input is a list or tensor or numpy array, converted to a numpy array, and we take the stack of values at those indices
    identity - Input is None, leave it unchanged.

    Examples for dim=0:
    if input_slice=0, tensor -> tensor[0]
    elif input_slice = (1, 5), tensor -> tensor[1:5]
    elif input_slice = (1, 5, 2), tensor -> tensor[1:5:2] (ie indexing with [1, 3])
    elif input_slice = [1, 4, 5], tensor -> tensor[[1, 4, 5]] (ie changing the first axis to have length 3, and taking the indices 1, 4, 5 out).
    elif input_slice is a Tensor, same as list - Tensor is assumed to be a 1D list of indices.
    """

    slice: Union[int, slice, np.ndarray]

    def __init__(
        self,
        input_slice: SliceInput = None,
    ):
        """
        Modular component for slicing tensors. Can be used to slice a tensor along a given dimension, or to index into a tensor along a given dimension.

        Args:
            input_slice (SliceInput): The slice to apply. Can be an int, a tuple, a list, a torch.Tensor, or None. If None, do nothing.

        Raises:
            ValueError: If the input_slice is not one of the above types.
        """
        if isinstance(input_slice, tuple):
            self.slice = slice(*input_slice)
            self.mode = "slice"
        elif isinstance(input_slice, int):
            self.slice = input_slice
            self.mode = "int"
        elif isinstance(input_slice, slice):
            self.slice = input_slice
            self.mode = "slice"
        elif type(input_slice) in [list, torch.Tensor, np.ndarray]:
            self.slice = to_numpy(input_slice)
            self.mode = "array"
        elif input_slice is None:
            self.slice = slice(None)
            self.mode = "identity"
        else:
            raise ValueError(f"Invalid input_slice {input_slice}")

    def apply(
        self,
        tensor: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """
        Takes in a tensor and a slice, and applies the slice to the given dimension (supports positive and negative dimension syntax). Returns the sliced tensor.

        Args:
            tensor (torch.Tensor): The tensor to slice.
            dim (int, optional): The dimension to slice along. Supports positive and negative dimension syntax.

        Returns:
            torch.Tensor: The sliced tensor.
        """
        ndim = tensor.ndim
        slices = [slice(None)] * ndim
        slices[dim] = self.slice  # type: ignore
        return tensor[tuple(slices)]

    def indices(
        self,
        max_ctx: Optional[int] = None,
    ) -> Union[np.ndarray, np.int32, np.int64]:
        """
        Returns the indices when this slice is applied to an axis of size max_ctx. Returns them as a numpy array, for integer slicing it is eg array([4])

        Args:
            max_ctx (int, optional): The size of the axis to slice. Only used if the slice is not an integer.

        Returns:
            np.ndarray: The indices that this slice will select.

        Raises:
            ValueError: If the slice is not an integer and max_ctx is not specified.
        """
        if self.mode == "int":
            return np.array([self.slice], dtype=np.int64)
        if max_ctx is None:
            raise ValueError("max_ctx must be specified if slice is not an integer")
        return np.arange(max_ctx, dtype=np.int64)[self.slice]

    def __repr__(
        self,
    ) -> str:
        return f"Slice: {self.slice} Mode: {self.mode} "

    @classmethod
    def unwrap(
        cls,
        slice_input: Union["Slice", SliceInput],
    ) -> "Slice":
        """
        Takes a Slice-like input and converts it into a Slice, if it is not already.

        Args:
            slice_input (Union[Slice, SliceInput]): The input to turn into a Slice.

        Returns:
            Slice: A Slice object.
        """
        if not isinstance(slice_input, Slice):
            if isinstance(
                slice_input, int
            ):  # slicing with an int collapses the dimension so this stops the pos dimension from collapsing
                slice_input = [slice_input]
            slice_input = Slice(slice_input)
        return slice_input


def get_act_name(
    name: str,
    layer: Optional[Union[int, str]] = None,
    layer_type: Optional[str] = None,
):
    """
    Helper function to convert shorthand to an activation name. Pretty hacky, intended to be useful for short feedback
    loop hacking stuff together, more so than writing good, readable code. But it is deterministic!

    Returns a name corresponding to an activation point in a TransformerLens model.

    Args:
         name (str): Takes in the name of the activation. This can be used to specify any activation name by itself.
         The code assumes the first sequence of digits passed to it (if any) is the layer number, and anything after
         that is the layer type.

         Given only a word and number, it leaves layer_type as is.
         Given only a word, it leaves layer and layer_type as is.

         Examples:
             get_act_name('embed') = get_act_name('embed', None, None)
             get_act_name('k6') = get_act_name('k', 6, None)
             get_act_name('scale4ln1') = get_act_name('scale', 4, 'ln1')

         layer (int, optional): Takes in the layer number. Used for activations that appear in every block.

         layer_type (string, optional): Used to distinguish between activations that appear multiple times in one block.

    Full Examples:

    get_act_name('k', 6, 'a')=='blocks.6.attn.hook_k'
    get_act_name('pre', 2)=='blocks.2.mlp.hook_pre'
    get_act_name('embed')=='hook_embed'
    get_act_name('normalized', 27, 'ln2')=='blocks.27.ln2.hook_normalized'
    get_act_name('k6')=='blocks.6.attn.hook_k'
    get_act_name('scale4ln1')=='blocks.4.ln1.hook_scale'
    get_act_name('pre5')=='blocks.5.mlp.hook_pre'
    """
    if ("." in name or name.startswith("hook_")) and layer is None and layer_type is None:
        # If this was called on a full name, just return it
        return name
    match = re.match(r"([a-z]+)(\d+)([a-z]?.*)", name)
    if match is not None:
        name, layer, layer_type = match.groups(0)  # type: ignore

    layer_type_alias = {
        "a": "attn",
        "m": "mlp",
        "b": "",
        "block": "",
        "blocks": "",
        "attention": "attn",
    }

    act_name_alias = {
        "attn": "pattern",
        "attn_logits": "attn_scores",
        "key": "k",
        "query": "q",
        "value": "v",
        "mlp_pre": "pre",
        "mlp_mid": "mid",
        "mlp_post": "post",
    }

    layer_norm_names = ["scale", "normalized"]

    if name in act_name_alias:
        name = act_name_alias[name]

    full_act_name = ""
    if layer is not None:
        full_act_name += f"blocks.{layer}."
    if name in [
        "k",
        "v",
        "q",
        "z",
        "rot_k",
        "rot_q",
        "result",
        "pattern",
        "attn_scores",
    ]:
        layer_type = "attn"
    elif name in ["pre", "post", "mid", "pre_linear"]:
        layer_type = "mlp"
    elif layer_type in layer_type_alias:
        layer_type = layer_type_alias[layer_type]

    if layer_type:
        full_act_name += f"{layer_type}."
    full_act_name += f"hook_{name}"

    if name in layer_norm_names and layer is None:
        full_act_name = f"ln_final.{full_act_name}"
    return full_act_name


def remove_batch_dim(tensor: Float[torch.Tensor, "1 ..."]) -> Float[torch.Tensor, "..."]:
    """
    Removes the first dimension of a tensor if it is size 1, otherwise returns the tensor unchanged
    """
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)
    else:
        return tensor


def test_prompt(
    prompt: str,
    answer: Union[str, list[str]],
    model,  # Can't give type hint due to circular imports
    prepend_space_to_answer: bool = True,
    print_details: bool = True,
    prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
    top_k: int = 10,
) -> None:
    """Test if the Model Can Give the Correct Answer to a Prompt.

    Intended for exploratory analysis. Prints out the performance on the answer (rank, logit, prob),
    as well as the top k tokens. Works for multi-token prompts and multi-token answers.

    Warning:

    This will print the results (it does not return them).

    Examples:

    >>> from transformer_lens import HookedTransformer, utils
    >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
    Loaded pretrained model tiny-stories-1M into HookedTransformer

    >>> prompt = "Why did the elephant cross the"
    >>> answer = "road"
    >>> utils.test_prompt(prompt, answer, model)
    Tokenized prompt: ['<|endoftext|>', 'Why', ' did', ' the', ' elephant', ' cross', ' the']
    Tokenized answer: [' road']
    Performance on answer token:
    Rank: 2        Logit: 14.24 Prob:  3.51% Token: | road|
    Top 0th token. Logit: 14.51 Prob:  4.59% Token: | ground|
    Top 1th token. Logit: 14.41 Prob:  4.18% Token: | tree|
    Top 2th token. Logit: 14.24 Prob:  3.51% Token: | road|
    Top 3th token. Logit: 14.22 Prob:  3.45% Token: | car|
    Top 4th token. Logit: 13.92 Prob:  2.55% Token: | river|
    Top 5th token. Logit: 13.79 Prob:  2.25% Token: | street|
    Top 6th token. Logit: 13.77 Prob:  2.21% Token: | k|
    Top 7th token. Logit: 13.75 Prob:  2.16% Token: | hill|
    Top 8th token. Logit: 13.64 Prob:  1.92% Token: | swing|
    Top 9th token. Logit: 13.46 Prob:  1.61% Token: | park|
    Ranks of the answer tokens: [(' road', 2)]

    Args:
        prompt:
            The prompt string, e.g. "Why did the elephant cross the".
        answer:
            The answer, e.g. "road". Note that if you set prepend_space_to_answer to False, you need
            to think about if you have a space before the answer here (as e.g. in this example the
            answer may really be " road" if the prompt ends without a trailing space). If this is a
            list of strings, then we only look at the next-token completion, and we compare them all
            as possible model answers.
        model:
            The model.
        prepend_space_to_answer:
            Whether or not to prepend a space to the answer. Note this will only ever prepend a
            space if the answer doesn't already start with one.
        print_details:
            Print the prompt (as a string but broken up by token), answer and top k tokens (all
            with logit, rank and probability).
        prepend_bos:
            Overrides self.cfg.default_prepend_bos if set. Whether to prepend
            the BOS token to the input (applicable when input is a string). Models generally learn
            to use the BOS token as a resting place for attention heads (i.e. a way for them to be
            "turned off"). This therefore often improves performance slightly.
        top_k:
            Top k tokens to print details of (when print_details is set to True).

    Returns:
        None (just prints the results directly).
    """
    answers = [answer] if isinstance(answer, str) else answer
    n_answers = len(answers)
    using_multiple_answers = n_answers > 1

    if prepend_space_to_answer:
        answers = [answer if answer.startswith(" ") else " " + answer for answer in answers]

    # GPT-2 often treats the first token weirdly, so lets give it a resting position
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    answer_tokens = model.to_tokens(answers, prepend_bos=False)

    # If we have multiple answers, we're only allowed a single token generation
    if using_multiple_answers:
        answer_tokens = answer_tokens[:, :1]

    # Deal with case where answers is a list of strings
    prompt_tokens = prompt_tokens.repeat(answer_tokens.shape[0], 1)
    tokens = torch.cat((prompt_tokens, answer_tokens), dim=1)

    prompt_str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
    answer_str_tokens_list = [model.to_str_tokens(answer, prepend_bos=False) for answer in answers]
    prompt_length = len(prompt_str_tokens)
    answer_length = 1 if using_multiple_answers else len(answer_str_tokens_list[0])
    if print_details:
        print("Tokenized prompt:", prompt_str_tokens)
        if using_multiple_answers:
            print("Tokenized answers:", answer_str_tokens_list)
        else:
            print("Tokenized answer:", answer_str_tokens_list[0])
    logits = model(tokens)
    probs = logits.softmax(dim=-1)
    answer_ranks = []

    for index in range(prompt_length, prompt_length + answer_length):
        # Get answer tokens for this sequence position
        answer_tokens = tokens[:, index]
        answer_str_tokens = [a[index - prompt_length] for a in answer_str_tokens_list]
        # Offset by 1 because models predict the NEXT token
        token_probs = probs[:, index - 1]
        sorted_token_probs, sorted_token_positions = token_probs.sort(descending=True)
        answer_token_ranks = sorted_token_positions.argsort(-1)[
            range(n_answers), answer_tokens.cpu()
        ].tolist()
        answer_ranks.append(
            [
                (answer_str_token, answer_token_rank)
                for answer_str_token, answer_token_rank in zip(
                    answer_str_tokens, answer_token_ranks
                )
            ]
        )
        if print_details:
            # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
            # rprint gives rich text printing
            rprint(
                f"Performance on answer token{'s' if n_answers > 1 else ''}:\n"
                + "\n".join(
                    [
                        f"[b]Rank: {answer_token_ranks[i]: <8} Logit: {logits[i, index-1, answer_tokens[i]].item():5.2f} Prob: {token_probs[i, answer_tokens[i]].item():6.2%} Token: |{answer_str_tokens[i]}|[/b]"
                        for i in range(n_answers)
                    ]
                )
            )
            for i in range(top_k):
                print(
                    f"Top {i}th token. Logit: {logits[0, index-1, sorted_token_positions[0, i]].item():5.2f} Prob: {sorted_token_probs[0, i].item():6.2%} Token: |{model.to_string(sorted_token_positions[0, i])}|"
                )

    # If n_answers = 1 then unwrap answer ranks, so printed output matches original version of function
    if not using_multiple_answers:
        single_answer_ranks = [r[0] for r in answer_ranks]
        rprint(f"[b]Ranks of the answer tokens:[/b] {single_answer_ranks}")
    else:
        rprint(f"[b]Ranks of the answer tokens:[/b] {answer_ranks}")


def transpose(tensor: Float[torch.Tensor, "... a b"]) -> Float[torch.Tensor, "... b a"]:
    """
    Utility to swap the last two dimensions of a tensor, regardless of the number of leading dimensions
    """
    return tensor.transpose(-1, -2)


def composition_scores(
    left: "FactoredMatrix", right: "FactoredMatrix", broadcast_dims=True
) -> Union[
    Float[torch.Tensor, "*leading_dims"],
    Float[torch.Tensor, "*leading_dims_left_and_right"],
]:
    """
    See `HookedTransformer.all_composition_scores` for documentation.
    """
    if broadcast_dims:
        r_leading = right.ndim - 2
        l_leading = left.ndim - 2
        for i in range(l_leading):
            right = right.unsqueeze(i)
        for i in range(r_leading):
            left = left.unsqueeze(i + l_leading)
    assert (
        left.rdim == right.ldim
    ), f"Composition scores require left.rdim==right.ldim, shapes were left: {left.shape}, right:{right.shape}"

    new_right = right.collapse_r()
    new_left = left.collapse_l()
    r_norms = new_right.norm(dim=[-2, -1])
    l_norms = new_left.norm(dim=[-2, -1])
    comp_norms = (new_left @ new_right).norm(dim=[-2, -1])
    return comp_norms / r_norms / l_norms


def get_dataset(dataset_name: str, **kwargs) -> Dataset:
    """
    Returns a small HuggingFace dataset, for easy testing and exploration. Accesses several convenience datasets with 10,000 elements (dealing with the enormous 100GB - 2TB datasets is a lot of effort!). Note that it returns a dataset (ie a dictionary containing all the data), *not* a DataLoader (iterator over the data + some fancy features). But you can easily convert it to a DataLoader.

    Each dataset has a 'text' field, which contains the relevant info, some also have several meta data fields

    Kwargs will be passed to the huggingface dataset loading function, e.g. "data_dir"

    Possible inputs:
    * openwebtext (approx the GPT-2 training data https://huggingface.co/datasets/openwebtext)
    * pile (The Pile, a big mess of tons of diverse data https://pile.eleuther.ai/)
    * c4 (Colossal, Cleaned, Common Crawl - basically openwebtext but bigger https://huggingface.co/datasets/c4)
    * code (Codeparrot Clean, a Python code dataset https://huggingface.co/datasets/codeparrot/codeparrot-clean )
    * c4_code (c4 + code - the 20K data points from c4-10k and code-10k. This is the mix of datasets used to train my interpretability-friendly models, though note that they are *not* in the correct ratio! There's 10K texts for each, but about 22M tokens of code and 5M tokens of C4)
    * wiki (Wikipedia, generated from the 20220301.en split of https://huggingface.co/datasets/wikipedia )
    """
    dataset_aliases = {
        "openwebtext": "stas/openwebtext-10k",
        "owt": "stas/openwebtext-10k",
        "pile": "NeelNanda/pile-10k",
        "c4": "NeelNanda/c4-10k",
        "code": "NeelNanda/code-10k",
        "python": "NeelNanda/code-10k",
        "c4_code": "NeelNanda/c4-code-20k",
        "c4-code": "NeelNanda/c4-code-20k",
        "wiki": "NeelNanda/wiki-10k",
    }
    if dataset_name in dataset_aliases:
        dataset = load_dataset(dataset_aliases[dataset_name], split="train", **kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset


def is_square(x: torch.Tensor) -> bool:
    """Checks if `x` is a square matrix."""
    return x.ndim == 2 and x.shape[0] == x.shape[1]


def is_lower_triangular(x: torch.Tensor) -> bool:
    """Checks if `x` is a lower triangular matrix."""
    if not is_square(x):
        return False
    return x.equal(x.tril())


def check_structure(t1: torch.Tensor, t2: torch.Tensor, *, verbose: bool = False) -> None:
    """Validate that the two square tensors have the same structure, i.e.,
    that the directionality of comparisons points in the same directions both
    row-wise and column-wise.

    This function is not used anywhere in the code right now, just for debugging tests.
    """
    assert t1.ndim == 2
    assert t1.shape == t2.shape
    n_rows, n_cols = cast(Tuple[int, int], t1.shape)

    if verbose:
        print("Checking rows")
    row_mismatch = []
    for row_i in range(n_rows - 1):
        t1_result = t1[row_i].ge(t1[row_i + 1])
        t2_result = t2[row_i].ge(t2[row_i + 1])
        if any(t1_result != t2_result):
            row_mismatch.append(row_i)
            if verbose:
                print(f"\trows {row_i}:{row_i + 1}")
                print(f"\tt1: {t1_result.tolist()}")
                print(f"\tt2: {t2_result.tolist()}")

    if verbose:
        print("Checking columns")
    col_mismatch = []
    for col_i in range(n_cols - 1):
        t1_result = t1[:, col_i].ge(t1[:, col_i + 1])
        t2_result = t2[:, col_i].ge(t2[:, col_i + 1])
        if any(t1_result != t2_result):
            col_mismatch.append(col_i)
            if verbose:
                print(f"\trows {col_i}:{col_i + 1}")
                print(f"\tt1: {t1_result.tolist()}")
                print(f"\tt2: {t2_result.tolist()}")
    if not row_mismatch and not col_mismatch:
        print("PASSED")
    elif row_mismatch:
        print(f"row mismatch: {row_mismatch}")
    elif col_mismatch:
        print(f"column mismatch: {col_mismatch}")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Parse the PyTorch version to check if it's below version 2.0
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:
            return torch.device("mps")

    return torch.device("cpu")


def override_or_use_default_value(
    default_flag: Any,
    override: Optional[Any] = None,
) -> Any:
    """
    Determines which flag to return based on whether an overriding flag is provided.
    If a not-None overriding flag is provided, it is returned.
    Otherwise, the global flag is returned.
    """
    return override if override is not None else default_flag


def get_offset_position_ids(
    past_kv_pos_offset: int,
    attention_mask: Int[torch.Tensor, "batch offset_pos"],
) -> Int[torch.Tensor, "batch pos"]:
    """
    Returns the indices of non-padded tokens, offset by the position of the first attended token.
    """
    # shift the position ids so that the id at the the first attended token position becomes zero.
    # The position ids of the prepending pad tokens are shifted to -1.
    shifted_position_ids = attention_mask.cumsum(dim=1) - 1  # [batch, tokens_length]

    # Set the position ids of all prepending pad tokens to an arbitrary number (zero here)
    # just to avoid indexing errors.
    position_ids = shifted_position_ids.masked_fill(shifted_position_ids < 0, 0)
    return position_ids[:, past_kv_pos_offset:]  # [pos, batch]


def get_cumsum_along_dim(tensor, dim, reverse=False):
    """
    Returns the cumulative sum of a tensor along a given dimension.
    """
    if reverse:
        tensor = tensor.flip(dims=(dim,))
    cumsum = tensor.cumsum(dim=dim)
    if reverse:
        cumsum = cumsum.flip(dims=(dim,))
    return cumsum


def get_attention_mask(tokenizer, tokens: torch.Tensor, prepend_bos: bool) -> torch.Tensor:
    """
    Computes the attention mask for the tokenized input.
    NOTE: Only the leftmost leading pads (when `padding_side == left`)
    or rightmost trailing pads (when `padding_side == right`) are
    considered as real pad tokens that should not be attended.

    Args:
        tokenizer: The tokenizer used for tokenization.
        tokens (torch.Tensor): The tokenized input.
        prepend_bos (bool): If True, a BOS token is prepended to the input.

    Returns:
        torch.Tensor: The attention mask for the input.
    """

    # Initialize the attention mask with ones (indicating all tokens should be attended to)
    attention_mask = torch.ones_like(tokens)
    is_not_pad_token = tokens.ne(tokenizer.pad_token_id)

    if tokenizer.padding_side == "right":
        # Zero-out the rightmost trailing pad tokens
        is_trailing_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=True) == 0
        attention_mask[is_trailing_pad] = 0
    else:
        # Zero-out the leftmost leading pad tokens
        is_leading_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=False) == 0
        attention_mask[is_leading_pad] = 0

        # If the bos token is the same as the pad token,
        # the last token of the leftmost leading pad tokens is the bos token.
        # We need to set the attention mask for the bos token to 1.
        if prepend_bos and tokenizer.bos_token_id == tokenizer.pad_token_id:
            pad_bos_positions = is_leading_pad.sum(-1) - 1
            attention_mask[torch.arange(attention_mask.shape[0]), pad_bos_positions] = 1

    return attention_mask


def repeat_along_head_dimension(
    tensor: Float[torch.Tensor, "batch pos d_model"],
    n_heads: int,
    clone_tensor=True,
    # `einops.repeat` uses a view in torch, so we generally clone the tensor to avoid using shared storage for each head entry
):
    repeated_tensor = einops.repeat(
        tensor,
        "batch pos d_model -> batch pos n_heads d_model",
        n_heads=n_heads,
    )
    if clone_tensor:
        return repeated_tensor.clone()
    else:
        return repeated_tensor


def get_nested_attr(obj, attr_str):
    """
    Retrieves a nested attribute from an object based on a dot-separated string.

    For example, if `attr_str` is "a.b.c", this function will return `obj.a.b.c`.

    Args:
        obj (Any): The object from which to retrieve the attribute.
        attr_str (str): A dot-separated string representing the attribute hierarchy.

    Returns:
        Any: The value of the nested attribute.
    """
    attrs = attr_str.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_str, value):
    """
    Sets a nested attribute of an object based on a dot-separated string.

    For example, if `attr_str` is "a.b.c", this function will set the value of `obj.a.b.c` to `value`.

    Args:
        obj (Any): The object on which to set the attribute.
        attr_str (str): A dot-separated string representing the attribute hierarchy.
        value (Any): The value to set for the nested attribute.
    """
    attrs = attr_str.split(".")

    # Navigate to the deepest object containing the attribute to be set
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)

    # Set the nested attribute's value
    setattr(obj, attrs[-1], value)


class LocallyOverridenDefaults:
    """
    Context manager that allows temporary overriding of default values within a model.
    Once the context is exited, the default values are restored.

    WARNING: This context manager must be used for any function/method that directly accesses
    default values which may be overridden by the user using the function/method's arguments,
    e.g., `model.cfg.default_prepend_bos` and `model.tokenizer.padding_side` which can be
    overriden by `prepend_bos` and `padding_side` arguments, respectively, in the `to_tokens`.
    """

    def __init__(self, model, **overrides):
        """
        Initializes the context manager.

        Args:
            model (HookedTransformer): The model whose default values will be overridden.
            overrides (dict): Key-value pairs of properties to override and their new values.
        """
        self.model = model
        self.overrides = overrides

        # Dictionary defining valid defaults, valid values, and locations to find and store them
        self.values_with_defaults = {
            "prepend_bos": {
                "default_location": "model.cfg.default_prepend_bos",
                "valid_values": [USE_DEFAULT_VALUE, True, False],
                "skip_overriding": False,
                "default_value_to_restore": None,  # Will be set later
            },
            "padding_side": {
                "default_location": "model.tokenizer.padding_side",
                "valid_values": [USE_DEFAULT_VALUE, "left", "right"],
                "skip_overriding": model.tokenizer is None,  # Do not override if tokenizer is None
                "default_value_to_restore": None,  # Will be set later
            },
        }

        # Ensure provided overrides are defined in the dictionary above
        for override in overrides:
            assert override in self.values_with_defaults, (
                f"{override} is not a valid parameter to override. "
                f"Valid parameters are {self.values_with_defaults.keys()}."
            )

    def __enter__(self):
        """
        Override default values upon entering the context.
        """
        for property, override in self.overrides.items():
            info = self.values_with_defaults[property]
            if info["skip_overriding"]:
                continue  # Skip if overriding for this property is disabled

            # Ensure the override is a valid value
            valid_values = info["valid_values"]
            assert (
                override in valid_values  # type: ignore
            ), f"{property} must be one of {valid_values}, but got {override}."

            # Fetch current default and store it to restore later
            default_location = info["default_location"]
            default_value = get_nested_attr(self, default_location)
            info["default_value_to_restore"] = deepcopy(default_value)

            # Override the default value
            locally_overriden_value = override_or_use_default_value(default_value, override)
            set_nested_attr(self, default_location, locally_overriden_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore default values upon exiting the context.
        """
        for property in self.overrides:
            info = self.values_with_defaults[property]
            if info["skip_overriding"]:
                continue

            # Restore the default value from before the context was entered
            default_location = info["default_location"]
            default_value = info["default_value_to_restore"]
            set_nested_attr(self, default_location, default_value)


def get_tokenizer_with_bos(tokenizer):
    """
    Returns the tokenizer initialized with add_bos_token=True.
    Such a tokenizer should be set as the default tokenizer because the tokenization of some
    tokenizers like LlamaTokenizer are different when bos token is automatically/manually
    prepended.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to initialize with add_bos_token=True.

    Returns:
        AutoTokenizer: The tokenizer initialized with add_bos_token=True.
    """
    init_kwargs = deepcopy(tokenizer.init_kwargs)
    pretrained_model_name_or_path = init_kwargs.pop("name_or_path")
    add_bos_token = init_kwargs.pop("add_bos_token", None)
    if add_bos_token is None:
        add_bos_token = getattr(tokenizer, "add_bos_token", False)

    if add_bos_token:
        tokenizer_with_bos = tokenizer
    else:
        huggingface_token = os.environ.get("HF_TOKEN", None)
        tokenizer_with_bos = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            add_bos_token=True,
            token=huggingface_token,
            **init_kwargs,
        )

    return tokenizer_with_bos


def get_input_with_manually_prepended_bos(tokenizer, input):
    """
    Manually prepends the bos token to the input.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for prepending the bos token.
        input (Union[str, List[str]]): The input to prepend the bos token to.

    Returns:
        Union[str, List[str]]: The input with the bos token manually prepended.
    """
    if isinstance(input, str):
        input = tokenizer.bos_token + input
    else:
        input = [tokenizer.bos_token + string for string in input]
    return input


def get_tokens_with_bos_removed(tokenizer, tokens):
    """
    Removes the bos token from the beginning of each sequence in `tokens`.
    The last dimension of `tokens` must be the sequence length.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used to tokenize the input.
        tokens (torch.Tensor): The tokenized input.

    Returns:
        torch.Tensor: The tokenized input with the bos token removed.
    """
    if tokenizer.padding_side == "right":
        return tokens[..., 1:]

    else:
        bos_removed_shape = list(tokens.shape)
        bos_removed_shape[-1] -= 1

        if tokenizer.bos_token_id == tokenizer.pad_token_id:
            is_not_pad_token = tokens.ne(tokenizer.pad_token_id)
            is_leading_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=False) == 0
            real_bos_positions = is_leading_pad.sum(-1) - 1
        else:
            real_bos_positions = (tokens == tokenizer.bos_token_id).int().argmax(-1)

        tokens = tokens.scatter(dim=1, index=real_bos_positions.unsqueeze(-1), value=-100)
        return tokens[tokens != -100].view(*bos_removed_shape)


try:
    import pytest

    # Note: Docstring won't be tested with PyTest (it's ignored), as it thinks this is a regular unit
    # test (because its name is prefixed `test_`).
    pytest.mark.skip(test_prompt)
except ModuleNotFoundError:
    pass  # disregard if pytest not in env
