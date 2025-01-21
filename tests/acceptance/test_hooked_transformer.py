import gc
import os

import pandas as pd
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer
from transformer_lens.components import LayerNormPre
from transformer_lens.HookedTransformer import DTYPE_FROM_STRING
from transformer_lens.loading_from_pretrained import (
    OFFICIAL_MODEL_NAMES,
    get_official_model_name,
)
from transformer_lens.utils import clear_huggingface_cache

TINY_STORIES_MODEL_NAMES = [
    name for name in OFFICIAL_MODEL_NAMES if name.startswith("roneneldan/TinyStories")
]

PYTHIA_MODEL_NAMES = [name for name in OFFICIAL_MODEL_NAMES if name.startswith("EleutherAI/pythia")]

model_names = [
    "attn-only-demo",
    "gpt2-small",
    "opt-125m",
    "gpt-neo-125M",
    "stanford-gpt2-small-a",
    "solu-4l-old",
    "solu-6l",
    "attn-only-3l",
    "pythia",
    "gelu-2l",
    "othello-gpt",
    "tiny-stories-33M",
    "bloom-560m",
    "santacoder",
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "google/gemma-2b",
    "google/gemma-7b",
]
text = "Hello world!"
""" 
# Code to regenerate loss store
store = {}
for name in model_names:
    model = HookedTransformer.from_pretrained(name, device='cuda')
    loss = model(text,return_type="loss")
    store[name] = loss.item()
print(store)
"""
loss_store = {
    "attn-only-demo": 5.701841354370117,
    "gpt2-small": 5.331855773925781,
    "opt-125m": 6.159054279327393,
    "gpt-neo-125M": 4.900552272796631,
    "stanford-gpt2-small-a": 5.652035713195801,
    "solu-4l-old": 5.6021833419799805,
    "solu-6l": 5.7042999267578125,
    "attn-only-3l": 5.747507095336914,
    "pythia": 4.659344673156738,
    "gelu-2l": 6.501802444458008,
    "redwood_attn_2l": 10.530948638916016,
    "solu-1l": 5.256411552429199,
    "tiny-stories-33M": 12.203617095947266,
    "bloom-560m": 5.237126350402832,
}

no_processing = [
    ("solu-1l", 5.256411552429199),
    (
        "redwood_attn_2l",
        10.530948638916016,
    ),  # TODO can't be loaded with from_pretrained
]


@pytest.mark.parametrize("name,expected_loss", list(loss_store.items()))
def test_model(name, expected_loss):
    # Runs the model on short text and checks if the loss is as expected
    model = HookedTransformer.from_pretrained(name)
    loss = model(text, return_type="loss")
    assert (loss.item() - expected_loss) < 4e-5
    del model
    gc.collect()

    if "GITHUB_ACTIONS" in os.environ:
        clear_huggingface_cache()


def test_othello_gpt():
    # like test model but Othello GPT has a weird input format
    # so we need to test it separately

    model = HookedTransformer.from_pretrained("othello-gpt")
    sample_input = torch.tensor(
        [
            [
                # fmt: off
                20, 19, 18, 10, 2, 1, 27, 3, 41, 42, 34, 12, 4, 40, 11, 29, 43, 13, 48, 56, 33,
                39, 22, 44, 24, 5, 46, 6, 32, 36, 51, 58, 52, 60, 21, 53, 26, 31, 37, 9, 25, 38,
                23, 50, 45, 17, 47, 28, 35, 30, 54, 16, 59, 49, 57, 14, 15, 55, 7,
                # fmt: on
            ]
        ]
    )
    loss = model(sample_input, return_type="loss")
    expected_loss = 1.9079375267028809
    assert (loss.item() - expected_loss) < 4e-5


@pytest.mark.parametrize("name,expected_loss", no_processing)
def test_from_pretrained_no_processing(name, expected_loss):
    # Checks if manually overriding the boolean flags in from_pretrained
    # is equivalent to using from_pretrained_no_processing

    model_ref = HookedTransformer.from_pretrained_no_processing(name)
    model_ref_config = model_ref.cfg
    reff_loss = model_ref(text, return_type="loss")
    del model_ref
    model_override = HookedTransformer.from_pretrained(
        name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        refactor_factored_attn_matrices=False,
    )
    assert model_ref_config == model_override.cfg

    if name != "redwood_attn_2l":  # TODO can't be loaded with from_pretrained
        # Do the converse check, i.e. check that overriding boolean flags in
        # from_pretrained_no_processing is equivalent to using from_pretrained
        model_ref = HookedTransformer.from_pretrained(name)
        model_ref_config = model_ref.cfg
        reff_loss = model_ref(text, return_type="loss")
        del model_ref
        model_override = HookedTransformer.from_pretrained_no_processing(
            name,
            fold_ln=True,
            center_writing_weights=True,
            center_unembed=True,
            refactor_factored_attn_matrices=False,
        )
        assert model_ref_config == model_override.cfg

    # also check losses
    print(reff_loss.item())
    assert (reff_loss.item() - expected_loss) < 4e-5


def test_process_weights_inplace():
    """Check that process_weights_ works"""
    model = HookedTransformer.from_pretrained_no_processing("gpt2-small")
    model.process_weights_()
    loss = model.forward(text, return_type="loss")
    assert (loss.item() - loss_store["gpt2-small"]) < 4e-5
    assert isinstance(model.ln_final, LayerNormPre)


def test_from_pretrained_revision():
    """
    Check that the from_pretrained parameter `revision` (= git version) works
    """

    _ = HookedTransformer.from_pretrained("gpt2", revision="main")

    try:
        _ = HookedTransformer.from_pretrained("gpt2", revision="inexistent_branch_name")
    except:
        pass
    else:
        raise AssertionError("Should have raised an error")


def test_bloom_similarity_with_hf_model_with_kv_cache_activated():
    tf_model = HookedTransformer.from_pretrained(
        "bigscience/bloom-560m", default_prepend_bos=False, device="cpu"
    )
    hf_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
    hf_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

    output_tf = tf_model.generate(
        text, do_sample=False, use_past_kv_cache=True, verbose=False, max_new_tokens=10
    )
    output_hf_tokens = hf_model.generate(
        hf_tokenizer(text, return_tensors="pt").input_ids,
        do_sample=False,
        max_new_tokens=10,
    )
    output_hf_str = hf_tokenizer.decode(output_hf_tokens[0], skip_special_tokens=True)

    assert output_tf == output_hf_str


def check_norm_folding(
    model_name,
    hf_model=None,
    tokenizer=None,
    prompt="Hello, world!",
    device=None,
    dtype=None,
):
    """
    Checks that loading a model with Layer/RMS Norm folding enabled does not (significantly) change its outputs.

    Returns the maximum difference between the logits produced by the same model with and without norm folding enabled.

    Also asserts that this difference is within some tolerance, although this is deliberately set to a high value
    in order to account for lower precision models.
    """

    # If a device/dtype is not specified, and hf_model is provided, use its device/dtype
    # Otherwise, default to cuda (if available)/float32
    if device is None:
        if hf_model:
            device = hf_model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        if hf_model:
            dtype = hf_model.dtype
        else:
            dtype = "float32"

    folded_model = HookedTransformer.from_pretrained(
        model_name=model_name,
        hf_model=hf_model,
        device=device,
        tokenizer=tokenizer,
        dtype=dtype,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
    )
    tokens = folded_model.to_tokens(prompt)
    folded_logits = folded_model(tokens).detach()
    del folded_model
    torch.cuda.empty_cache()

    unfolded_model = HookedTransformer.from_pretrained(
        model_name=model_name,
        hf_model=hf_model,
        device=device,
        tokenizer=tokenizer,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    unfolded_logits = unfolded_model(tokens).detach()
    del unfolded_model
    torch.cuda.empty_cache()

    assert torch.allclose(
        torch.softmax(folded_logits, dim=-1),
        torch.softmax(unfolded_logits, dim=-1),
        atol=1e-2,
    )

    return torch.max(
        torch.abs(torch.softmax(folded_logits, dim=-1) - torch.softmax(unfolded_logits, dim=-1))
    )


def calculate_error(logits1, logits2):
    t1 = torch.softmax(logits1, dim=-1).to("cpu")
    t2 = torch.softmax(logits2, dim=-1).to("cpu")
    err = torch.abs(t1 - t2)
    return {
        "max": torch.max(err).item(),
        "mean": torch.mean(err).item(),
        "median": torch.median(err).item(),
        "std": torch.std(err).item(),
    }


def benchmark_model_options(
    model_name: str,
    hf_model=None,
    tokenizer=None,
    device="cuda",
    n_devices=1,
    dtype=torch.float16,
    cache_in_cpu=True,
):
    options = {
        "fold_ln": False,
        "center_writing_weights": False,
        "center_unembed": False,
        "fold_value_biases": False,
    }

    prompts = [
        "Hello, world!",
        "This is a test.",
        "What is it about?",
        "I don't know.",
    ]

    model_name = get_official_model_name(model_name)

    if hf_model is None:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=4).input_ids.to(
        device
    )

    # hf_model = hf_model.to(device)
    hf_logits = hf_model(tokens).logits.detach()
    hf_logits = hf_logits.to("cpu")

    if cache_in_cpu:
        hf_model = hf_model.to("cpu")
    else:
        del hf_model
        hf_model = None

    torch.cuda.empty_cache()
    gc.collect()

    results = {}

    # Check the error when all processing options are disabled
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        n_devices=n_devices,
        dtype=dtype,
        **options,
    )
    tl_logits = tl_model(tokens).detach().to("cpu")
    results["no_options"] = calculate_error(hf_logits, tl_logits)
    del tl_model, tl_logits
    torch.cuda.empty_cache()

    # Check the error when each processing option is enabled individually
    for option in options:
        gc.collect()
        new_options = options.copy()
        new_options[option] = True
        tl_model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            n_devices=n_devices,
            dtype=dtype,
            **new_options,
        )
        tl_logits = tl_model(tokens).detach().to("cpu")
        results[option] = calculate_error(hf_logits, tl_logits)

        del tl_model, tl_logits
        torch.cuda.empty_cache()
        gc.collect()

    # Check the error when all processing options are enabled
    all_options = {k: True for k, v in options.items()}
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        n_devices=n_devices,
        dtype=dtype,
        **all_options,
    )
    tl_logits = tl_model(tokens).detach().to("cpu")
    results["all_options"] = calculate_error(hf_logits, tl_logits)

    del tl_model, tl_logits

    del hf_model
    del tokens
    gc.collect()
    torch.cuda.empty_cache()

    return results


def benchmark_models(models, device="cuda", n_devices=1, cache_in_cpu=True):
    """
    Benchmark the error introduced by different options and data types for a list of models.
    :param models: A dict mapping model names to a list of dtypes to test
    """
    rows = []

    for model in models:
        dtypes = models[model]
        for dtype in dtypes:
            print(f"Testing {model} with dtype {dtype}")
            results = benchmark_model_options(
                model,
                device=device,
                n_devices=n_devices,
                dtype=DTYPE_FROM_STRING[dtype],
                cache_in_cpu=cache_in_cpu,
            )
            for option, result in results.items():
                rows.append({"model": model, "dtype": dtype, "options": option, **result})

    return pd.DataFrame(rows)


def check_similarity_with_hf_model(tl_model, hf_model, prompt="Hello, world!"):
    """
    Check that the TransformerLens model and the HuggingFace model
    give approximately the same results.

    The logits typically differ by a constant value, but check only the results
    after the softmax because this is what matters most.
    """
    tokens = tl_model.tokenizer.encode(prompt, return_tensors="pt")
    logits = tl_model(tokens, prepend_bos=False)
    hf_logits = hf_model(tokens).logits
    assert torch.allclose(
        torch.softmax(logits, dim=-1), torch.softmax(hf_logits, dim=-1), atol=1e-5
    )


def check_performance(tl_model, hf_model, margin):
    """
    Check that the TransformerLens model and the HuggingFace have
    approximately the same confidence in the expected answer.
    """
    prompt = " Unable"
    tokens = tl_model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    expected_token = tl_model.tokenizer.encode(" to")[
        0
    ]  # Assume this is the expected token to predict

    tl_logits = tl_model(tokens, prepend_bos=False)[0, -1].float()
    hf_logits = hf_model(tokens).logits[0, -1].float()
    tl_prob = torch.softmax(tl_logits, dim=-1)[expected_token].item()
    hf_prob = torch.softmax(hf_logits, dim=-1)[expected_token].item()
    assert tl_prob + margin > hf_prob


def check_dtype(dtype, margin, no_processing=False):
    """Check the loading and inferences for different dtypes."""
    for model_path in ["gpt2", "roneneldan/TinyStories-33M", "EleutherAI/pythia-70m"]:
        if no_processing:
            # For low precision, the processing is not advised.
            model = HookedTransformer.from_pretrained_no_processing(model_path, torch_dtype=dtype)
        else:
            model = HookedTransformer.from_pretrained(model_path, torch_dtype=dtype)

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        for layer_name, layer in model.state_dict().items():
            assert layer.dtype in [dtype, torch.bool] or "IGNORE" in layer_name

        check_performance(model, hf_model, margin)

        # Check that generate doesn't throw an error
        _ = model.generate("Hello, World!")

        del model
        del hf_model
        gc.collect()


@pytest.mark.skipif(
    torch.backends.mps.is_available() or not torch.cuda.is_available(),
    reason="some operations unsupported by MPS: https://github.com/pytorch/pytorch/issues/77754 or no GPU",
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_dtype_float(dtype):
    check_dtype(dtype, margin=5e-4)


@pytest.mark.skipif(
    torch.backends.mps.is_available() or not torch.cuda.is_available(),
    reason="bfloat16 unsupported by MPS: https://github.com/pytorch/pytorch/issues/78168 or no GPU",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision(dtype):
    """Check the 16 bits loading and inferences.
    Note that bfloat16 is generally preferred to float16 for ML due to numerical instabilities,
    and some float16 operations require having a GPU.
    bfloat16 can be used without GPU, but surprisingly it doesn't give the same results in this case.
    """
    check_dtype(dtype, margin=0.05, no_processing=True)


@torch.no_grad()
def test_pos_embed_hook():
    """
    Checks that pos embed hooks:
    - do not permanently change the pos embed
    - can be used to alter the pos embed for a specific batch element
    """
    model = HookedTransformer.from_pretrained("gpt2-small")
    initial_W_pos = model.W_pos.detach().clone()

    def remove_pos_embed(z, hook):
        z[:] = 0.0
        return z

    _ = model.run_with_hooks("Hello, world", fwd_hooks=[("hook_pos_embed", remove_pos_embed)])

    # Check that pos embed has not been permanently changed
    assert (model.W_pos == initial_W_pos).all()

    def edit_pos_embed(z, hook):
        sequence_length = z.shape[1]
        z[1, :] = 0.0
        # Check that the second batch element is zeroed
        assert (z[1, :] == 0.0).all()
        # Check that the first batch element is unchanged
        assert (z[0, :] == initial_W_pos[:sequence_length]).all()
        return z

    _ = model.run_with_hooks(
        ["Hello, world", "Goodbye, world"],
        fwd_hooks=[("hook_pos_embed", edit_pos_embed)],
    )


def test_all_tinystories_models_exist():
    for model in TINY_STORIES_MODEL_NAMES:
        try:
            AutoConfig.from_pretrained(model)
        except OSError:
            pytest.fail(
                f"Could not download model '{model}' from Huggingface."
                " Maybe the name was changed or the model has been removed."
            )


def test_all_pythia_models_exist():
    for model in PYTHIA_MODEL_NAMES:
        try:
            AutoConfig.from_pretrained(model)
        except OSError:
            pytest.fail(
                f"Could not download model '{model}' from Huggingface."
                " Maybe the name was changed or the model has been removed."
            )


@pytest.mark.parametrize(
    "input_type,return_type",
    [
        ("str", "input"),
        ("str", "str"),
        ("str", "tokens"),
        ("str", "embeds"),
        ("tokens", "input"),
        ("tokens", "str"),
        ("tokens", "tokens"),
        ("tokens", "embeds"),
        ("embeds", "input"),
        ("embeds", "str"),
        ("embeds", "tokens"),
        ("embeds", "embeds"),
    ],
)
def test_different_inputs_for_generation(
    input_type, return_type, print_output=False, max_new_tokens=3
):
    from typing import List

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hooked_llm = HookedTransformer.from_pretrained("gpt2", device=device)

    hooked_llm.eval()
    for text_input in [
        "What is the meaning of life?",
        ["AI will destroy world", "AI will save us"],
    ]:
        is_batched = False if isinstance(text_input, str) else True

        tokens_input = hooked_llm.to_tokens(text_input)
        embeddings_input = hooked_llm.embed(tokens_input)

        if input_type == "str":
            model_input = text_input
        elif input_type == "tokens":
            model_input = tokens_input
        elif input_type == "embeds":
            model_input = embeddings_input
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        output = hooked_llm.generate(
            input=model_input, max_new_tokens=max_new_tokens, return_type=return_type, verbose=False
        )

        if return_type == "str" or (return_type == "input" and input_type == "str"):
            if is_batched:
                assert isinstance(output, List), f"Expected list output but got {type(output)}"
                assert isinstance(
                    output[0], str
                ), f"Expected list of strings but got list of {type(output[0])}"
            else:
                assert isinstance(output, str), f"Expected string output but got {type(output)}"
        elif return_type == "tokens" or (return_type == "input" and input_type == "tokens"):
            assert isinstance(
                output, torch.Tensor
            ), f"Expected tensor output but got {type(output)}"
            assert output.ndim == 2, f"Expected 2D tensor but got {output.ndim}D"
        elif return_type == "embeds" or (return_type == "input" and input_type == "embeds"):
            assert isinstance(
                output, torch.Tensor
            ), f"Expected tensor output but got {type(output)}"
            assert output.ndim == 3, f"Expected 3D tensor but got {output.ndim}D"

        if print_output:
            print(f"Input type: {input_type}, return type: {return_type}, output:\n{output}")

    if print_output:
        print()
