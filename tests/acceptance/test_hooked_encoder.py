from typing import List

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch.testing import assert_close
from transformers import AutoTokenizer, BertForMaskedLM, BertForNextSentencePrediction

from transformer_lens import HookedEncoder, NextSentencePrediction

MODEL_NAME = "bert-base-cased"


@pytest.fixture(scope="module")
def our_bert():
    return HookedEncoder.from_pretrained(MODEL_NAME, device="cpu")


@pytest.fixture(scope="module")
def our_bert_nsp():
    return NextSentencePrediction.from_pretrained(MODEL_NAME, device="cpu")


@pytest.fixture(scope="module")
def huggingface_bert_mlm():
    return BertForMaskedLM.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def huggingface_bert_nsp():
    return BertForNextSentencePrediction.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def hello_world_tokens(tokenizer):
    return tokenizer("Hello, world!", return_tensors="pt")["input_ids"]


def test_full_model(our_bert, huggingface_bert_mlm, tokenizer):
    sequences = [
        "Hello, world!",
        "this is another sequence of tokens",
    ]
    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    huggingface_bert_out = huggingface_bert_mlm(input_ids, attention_mask=attention_mask).logits
    our_bert_out = our_bert(input_ids, one_zero_attention_mask=attention_mask)
    assert_close(huggingface_bert_out, our_bert_out, rtol=1.3e-6, atol=4e-5)


def test_embed_one_sentence(our_bert, huggingface_bert_mlm, hello_world_tokens):
    huggingface_embed = huggingface_bert_mlm.bert.embeddings
    our_embed = our_bert.embed

    huggingface_embed_out = huggingface_embed(hello_world_tokens)[0]
    our_embed_out = our_embed(hello_world_tokens).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_embed_two_sentences(our_bert, huggingface_bert_mlm, tokenizer):
    encoding = tokenizer("First sentence.", "Second sentence.", return_tensors="pt")
    input_ids = encoding["input_ids"]
    token_type_ids = encoding["token_type_ids"]

    huggingface_embed_out = huggingface_bert_mlm.bert.embeddings(
        input_ids, token_type_ids=token_type_ids
    )[0]
    our_embed_out = our_bert.embed(input_ids, token_type_ids=token_type_ids).squeeze(0)
    assert_close(huggingface_embed_out, our_embed_out)


def test_attention(our_bert, huggingface_bert_mlm, hello_world_tokens):
    huggingface_embed = huggingface_bert_mlm.bert.embeddings
    huggingface_attn = huggingface_bert_mlm.bert.encoder.layer[0].attention

    embed_out = huggingface_embed(hello_world_tokens)

    our_attn = our_bert.blocks[0].attn

    our_attn_out = our_attn(embed_out, embed_out, embed_out)
    huggingface_self_attn_out = huggingface_attn.self(embed_out)[0]
    huggingface_attn_out = huggingface_attn.output.dense(huggingface_self_attn_out)
    assert_close(our_attn_out, huggingface_attn_out)


def test_bert_block(our_bert, huggingface_bert_mlm, hello_world_tokens):
    huggingface_embed = huggingface_bert_mlm.bert.embeddings
    huggingface_block = huggingface_bert_mlm.bert.encoder.layer[0]

    embed_out = huggingface_embed(hello_world_tokens)

    our_block = our_bert.blocks[0]

    our_block_out = our_block(embed_out)
    huggingface_block_out = huggingface_block(embed_out)[0]
    assert_close(our_block_out, huggingface_block_out)


def test_bert_pooler(our_bert_nsp, huggingface_bert_nsp, hello_world_tokens):
    huggingface_bert_core_outputs = huggingface_bert_nsp.bert(hello_world_tokens).last_hidden_state

    our_pooler_out = our_bert_nsp.pooler(huggingface_bert_core_outputs)
    huggingface_pooler_out = huggingface_bert_nsp.bert.pooler(huggingface_bert_core_outputs)
    assert_close(our_pooler_out, huggingface_pooler_out)


def test_nsp_head(our_bert_nsp, huggingface_bert_nsp, hello_world_tokens):
    huggingface_bert_core_outputs = huggingface_bert_nsp.bert(hello_world_tokens).last_hidden_state

    our_pooler_out = our_bert_nsp.pooler(huggingface_bert_core_outputs)
    huggingface_pooler_out = huggingface_bert_nsp.bert.pooler(huggingface_bert_core_outputs)

    our_nsp_head_out = our_bert_nsp.nsp_head(our_pooler_out)
    huggingface_nsp_head_out = huggingface_bert_nsp.cls.seq_relationship(huggingface_pooler_out)

    assert_close(our_nsp_head_out, huggingface_nsp_head_out)


def test_mlm_head(our_bert, huggingface_bert_mlm, hello_world_tokens):
    huggingface_bert_core_outputs = huggingface_bert_mlm.bert(hello_world_tokens).last_hidden_state

    our_mlm_head_out = our_bert.mlm_head(huggingface_bert_core_outputs)
    huggingface_predictions_out = huggingface_bert_mlm.cls.predictions.transform(
        huggingface_bert_core_outputs
    )

    print((our_mlm_head_out - huggingface_predictions_out).abs().max())
    assert_close(our_mlm_head_out, huggingface_predictions_out, rtol=1.3e-3, atol=1e-5)


def test_unembed(our_bert, huggingface_bert_mlm, hello_world_tokens):
    huggingface_bert_core_outputs = huggingface_bert_mlm.bert(hello_world_tokens).last_hidden_state

    our_mlm_head_out = our_bert.mlm_head(huggingface_bert_core_outputs)
    our_unembed_out = our_bert.unembed(our_mlm_head_out)
    huggingface_predictions_out = huggingface_bert_mlm.cls.predictions(
        huggingface_bert_core_outputs
    )

    assert_close(our_unembed_out, huggingface_predictions_out, rtol=1.3e-6, atol=4e-5)


def test_run_with_cache_mlm(our_bert, huggingface_bert_mlm, hello_world_tokens):
    model = HookedEncoder.from_pretrained("bert-base-cased")
    logits, cache = model.run_with_cache(hello_world_tokens)

    # check that an arbitrary subset of the keys exist
    assert "embed.hook_embed" in cache
    assert "blocks.0.attn.hook_q" in cache
    assert "blocks.3.attn.hook_attn_scores" in cache
    assert "blocks.7.hook_resid_post" in cache
    assert "mlm_head.ln.hook_normalized" in cache


def test_run_with_cache_nsp():
    model = NextSentencePrediction.from_pretrained("bert-base-cased")
    sentences = ["She went to the grocery store.", "She bought a loaf of bread."]
    logits, cache = model.run_with_cache(sentences)

    # check that an arbitrary subset of the keys exist
    assert "embed.hook_embed" in cache
    assert "blocks.0.attn.hook_q" in cache
    assert "blocks.3.attn.hook_attn_scores" in cache
    assert "blocks.7.hook_resid_post" in cache
    assert "pooler.hook_pooler_out" in cache
    assert "nsp_head.hook_nsp_out" in cache


def test_from_pretrained_revision():
    """
    Check that the from_pretrained parameter `revision` (= git version) works
    """

    _ = HookedEncoder.from_pretrained(MODEL_NAME, revision="main")

    try:
        _ = HookedEncoder.from_pretrained(MODEL_NAME, revision="inexistent_branch_name")
    except:
        pass
    else:
        raise AssertionError("Should have raised an error")


@pytest.mark.skipif(
    torch.backends.mps.is_available() or not torch.cuda.is_available(),
    reason="bfloat16 unsupported by MPS: https://github.com/pytorch/pytorch/issues/78168 or no GPU",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision(dtype):
    """Check the 16 bits loading and inferences."""
    model = HookedEncoder.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    assert model.W_K.dtype == dtype

    _ = model(model.tokenizer("Hello, world", return_tensors="pt")["input_ids"])


def _get_predictions(
    logits: Float[torch.Tensor, "batch pos d_vocab"], positions: List[int], tokenizer
):
    logits_at_position = logits.squeeze(0)[positions]
    predicted_tokens = F.softmax(logits_at_position, dim=-1).argmax(dim=-1)
    return tokenizer.batch_decode(predicted_tokens)


def test_predictions(our_bert, huggingface_bert_mlm, tokenizer):
    input_ids = tokenizer("The [MASK] sat on the mat", return_tensors="pt")["input_ids"]

    our_bert_out = our_bert(input_ids)
    our_prediction = _get_predictions(our_bert_out, [2], tokenizer)

    huggingface_bert_out = huggingface_bert_mlm(input_ids).logits
    huggingface_prediction = _get_predictions(huggingface_bert_out, [2], tokenizer)

    assert our_prediction == huggingface_prediction


def test_predictions_from_forward_function_mlm(our_bert, huggingface_bert_mlm, tokenizer):
    prompt = "The [MASK] sat on the mat"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    our_prediction = our_bert(prompt, return_type="predictions")

    huggingface_bert_out = huggingface_bert_mlm(input_ids).logits
    huggingface_prediction = _get_predictions(huggingface_bert_out, [2], tokenizer)[
        0
    ]  # prediction is returned as a list

    assert our_prediction == huggingface_prediction


def test_predictions_from_forward_function_nsp(our_bert_nsp, huggingface_bert_nsp, tokenizer):
    sentence_a = "The cat sat on the mat"
    sentence_b = "The dog sat on the mat"
    input_ids = tokenizer(sentence_a, sentence_b, return_tensors="pt")["input_ids"]

    our_prediction = our_bert_nsp([sentence_a, sentence_b])
    huggingface_bert_out = huggingface_bert_nsp(input_ids).logits

    our_logprobs = our_prediction.log_softmax(dim=-1)
    hugging_face_logprobs = huggingface_bert_out.log_softmax(dim=-1)

    our_prediction = our_logprobs.argmax(dim=-1)
    huggingface_prediction = hugging_face_logprobs.argmax(dim=-1)
    print("Our prediction: ", our_prediction)
    print("Huggingface prediction: ", huggingface_prediction)
    assert our_prediction == huggingface_prediction


def test_input_list_of_strings_mlm(our_bert, huggingface_bert_mlm, tokenizer):
    prompts = ["The [MASK] sat on the mat", "She [MASK] to the store", "The dog [MASK] the ball"]
    encodings = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True)
    our_bert_out = our_bert(prompts)

    huggingface_bert_out = huggingface_bert_mlm(**encodings).logits

    assert_close(our_bert_out, huggingface_bert_out, rtol=1.3e-6, atol=4e-5)


def test_wrong_sentence_input_causes_error_nsp(our_bert_nsp):
    sentence_a = "The cat sat on the mat"
    sentence_b = "The dog sat on the mat"
    sentence_c = "The bird sat on the mat"
    sentences = [sentence_a, sentence_b, sentence_c]
    with pytest.raises(ValueError):
        our_bert_nsp(sentences)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_cuda(hello_world_tokens):
    model = HookedEncoder.from_pretrained(MODEL_NAME)
    model(hello_world_tokens)
