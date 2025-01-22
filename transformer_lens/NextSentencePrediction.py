"""Next Sentence Prediction.

Contains a BERT style model specifically for Next Sentence Prediction. This is separate from 
:class:`transformer_lens.HookedTransformer` because it has a significantly different architecture 
to e.g. GPT style transformers.
"""


from typing import Dict, List, Optional, Tuple, Union, overload

import torch
from jaxtyping import Float, Int
from typing_extensions import Literal, override

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import BertNSPHead, BertPooler
from transformer_lens.HookedEncoder import HookedEncoder


class NextSentencePrediction(HookedEncoder):
    """A BERT-style model for Next Sentence Prediction (NSP) that extends HookedEncoder.

    This class implements a BERT model specifically designed for the Next Sentence Prediction task,
    where the model predicts whether two input sentences naturally follow each other in the original text.
    It inherits from HookedEncoder and adds NSP-specific components like the NSP head and pooler layer.

    The model processes pairs of sentences and outputs either logits or human-readable predictions
    indicating whether the sentences are sequential. String inputs are automatically tokenized with
    appropriate token type IDs to distinguish between the two sentences.

    Note:
        This model expects inputs to be provided as pairs of sentences. Single sentence inputs
        or inputs without proper sentence separation will raise errors.
    """

    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__(cfg, tokenizer, move_to_device, **kwargs)
        self.nsp_head = BertNSPHead(self.cfg)
        self.pooler = BertPooler(self.cfg)

        if move_to_device:
            self.to(self.cfg.device)

        self.setup()

    @override
    def to_tokens(
        self,
        input: Union[str, List[str]],
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Tuple[
        Int[torch.Tensor, "batch pos"],
        Int[torch.Tensor, "batch pos"],
        Int[torch.Tensor, "batch pos"],
    ]:
        """Converts a string to a tensor of tokens.
        Taken mostly from the HookedTransformer implementation, but does not support default padding
        sides or prepend_bos.
        Args:
            input (Union[str, List[str]]): The input to tokenize.
            move_to_device (bool): Whether to move the output tensor of tokens to the device the
                model lives on. Defaults to True
            truncate (bool): If the output tokens are too long, whether to truncate the output
                tokens to the model's max context window. Does nothing for shorter inputs.
                Defaults to True.
        """

        if isinstance(input, str) or len(input) != 2:
            raise ValueError(
                "Next sentence prediction task requires exactly two sentences, please provide a list of strings with each sentence as an element."
            )

        # We need to input the two sentences separately for NSP
        encodings = self.tokenizer(
            input[0],
            input[1],
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )

        tokens = encodings.input_ids

        if move_to_device:
            tokens = tokens.to(self.cfg.device)
            token_type_ids = encodings.token_type_ids.to(self.cfg.device)
            attention_mask = encodings.attention_mask.to(self.cfg.device)

        return tokens, token_type_ids, attention_mask

    @override  # type: ignore[override]
    def forward(
        self,
        input: Union[
            List[str],
            Int[torch.Tensor, "batch pos"],
        ],
        return_type: Optional[str] = "logits",
        token_type_ids: Optional[Int[torch.Tensor, "batch pos"]] = None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Union[Float[torch.Tensor, "batch 2"], str]]:
        """Forward pass through the NextSentencePrediction module. Performs Next Sentence Prediction on a pair of sentences.

        Args:
            input: The input to process. Can be one of:
                - List[str]: A list of two strings representing the two sentences NSP
                                should be performed on
                - torch.Tensor: Input tokens as integers with shape (batch, position)
            return_type: Optional[str]: The type of output to return. Can be one of:
                - None: Return nothing, don't calculate logits
                - 'logits': Return logits tensor
                - 'predictions': Return human-readable predictions
            token_type_ids: Optional[torch.Tensor]: Binary ids indicating whether a token belongs
                to sequence A or B. For example, for two sentences:
                "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be
                [0, 0, ..., 0, 1, ..., 1, 1]. `0` represents tokens from Sentence A,
                `1` from Sentence B. If not provided, BERT assumes a single sequence input.
                This parameter gets inferred from the the tokenizer if input is a string or list of strings.
                Shape is (batch_size, sequence_length).
            one_zero_attention_mask: Optional[torch.Tensor]: A binary mask which indicates
                which tokens should be attended to (1) and which should be ignored (0).
                Primarily used for padding variable-length sentences in a batch.
                For instance, in a batch with sentences of differing lengths, shorter
                sentences are padded with 0s on the right. If not provided, the model
                assumes all tokens should be attended to.
                This parameter gets inferred from the tokenizer if input is a string or list of strings.
                Shape is (batch_size, sequence_length).

        Returns:
            Optional[torch.Tensor]: Depending on return_type:
                - None: Returns None if return_type is None
                - torch.Tensor: Returns logits if return_type is 'logits' (or if return_type is not explicitly provided)
                    - Shape is (batch_size, 2)
                - str or List[str]: Returns string indicating if sentences are sequential if return_type is 'predictions'

        Raises:
            ValueError: If using NSP task without proper input format or token_type_ids
            AssertionError: If using string input without a tokenizer
        """

        if token_type_ids == None and isinstance(input, torch.Tensor):
            raise ValueError(
                "You are using the NSP task without specifying token_type_ids."
                "This means that the model will treat the input as a single sequence which will lead to incorrect results."
                "Please provide token_type_ids or use a string input."
            )

        resid, _ = self.encoder_output(input, token_type_ids, one_zero_attention_mask)

        # NSP requires pooling (for more information see BertPooler)
        resid = self.pooler(resid)
        logits = self.nsp_head(resid)

        if return_type == "predictions":
            logprobs = logits.log_softmax(dim=-1)
            predictions = [
                "The sentences are sequential",
                "The sentences are NOT sequential",
            ]
            return predictions[logprobs.argmax(dim=-1).item()]

        elif return_type == None:
            return None

        return logits

    @override
    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch 2"], ActivationCache,]:
        ...

    @override
    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch 2"], Dict[str, torch.Tensor],]:
        ...

    @override
    def run_with_cache(
        self,
        *model_args,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[Float[torch.Tensor, "batch 2"], Union[ActivationCache, Dict[str, torch.Tensor]],]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True,
        this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods,
        otherwise it will return a dictionary of activations as in HookedRootModule.
        This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super(HookedEncoder, self).run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )

        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict
