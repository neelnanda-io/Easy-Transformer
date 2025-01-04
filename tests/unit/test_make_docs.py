"""Make Docs Tests."""

import pytest

from docs.make_docs import get_config, get_model_info, get_property
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def test_get_config():
    """Test get config with attn-only-1l model."""
    config: HookedTransformerConfig = get_config("attn-only-1l")
    assert config.attn_only is True


def test_get_property():
    """Test get property with attn-only-1l model."""
    act_fn = get_property("act_fn", "attn-only-1l")
    assert act_fn == "attn_only"

    n_params = get_property("n_params", "attn-only-1l")
    assert n_params == "1.0M"

    n_layers = get_property("n_layers", "attn-only-1l")
    assert n_layers == 1

    d_model = get_property("d_model", "attn-only-1l")
    assert d_model == 512

    n_heads = get_property("n_heads", "attn-only-1l")
    assert n_heads == 8

    n_ctx = get_property("n_ctx", "attn-only-1l")
    assert n_ctx == 1024

    d_vocab = get_property("d_vocab", "attn-only-1l")
    assert d_vocab == 48262

    d_head = get_property("d_head", "attn-only-1l")
    assert d_head == 64

    d_mlp = get_property("d_mlp", "attn-only-1l")
    assert d_mlp == 2048

    n_key_value_heads = get_property("n_key_value_heads", "attn-only-1l")
    assert n_key_value_heads is None

    # Test an unknown property
    with pytest.raises(KeyError):
        get_property("unknown_property", "attn-only-1l")


def test_get_model_info():
    get_model_info("attn-only-1l")
