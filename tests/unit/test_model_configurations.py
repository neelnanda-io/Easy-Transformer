from functools import lru_cache

import pytest

from transformer_lens import loading
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


@lru_cache(maxsize=None)
def get_cached_config(model_name: str) -> HookedTransformerConfig:
    """Retrieve the configuration of a pretrained model.

    Args:
        model_name (str): Name of the pretrained model.

    Returns:
        HookedTransformerConfig: Configuration of the pretrained model.
    """
    return loading.get_pretrained_model_config(model_name)


@pytest.mark.parametrize("model_name", loading.DEFAULT_MODEL_ALIASES)
def test_model_configurations(model_name: str):
    """Tests that all of the model configurations are in fact loaded (e.g. are not None)."""
    assert get_cached_config(model_name) is not None, f"Configuration for {model_name} is None"
