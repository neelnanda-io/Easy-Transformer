import torch

from transformer_lens.components import AbstractAttention, RotaryEmbedding
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def test_create_alibi_slope():
    n_ctx = 100

    # Expected result computed non-vectorized way
    expected = torch.zeros((n_ctx, n_ctx))
    for row in range(n_ctx):
        for col in range(n_ctx):
            expected[row, col] = float(min(col - row, 0))

    # Check against the method's vectorized version
    result = AbstractAttention.create_alibi_slope(n_ctx)
    assert torch.allclose(expected, result)


def test_create_alibi_bias():
    n_heads = 2
    n_ctx = 4

    result = AbstractAttention.create_alibi_bias(n_heads, n_ctx, torch.device("cpu"))

    for matrix in result:
        n_row, n_col = matrix.size()
        slope = -matrix[1, 0]
        # Check if upper triangle is all zeros
        assert torch.equal(torch.triu(matrix), torch.zeros_like(matrix))

        ref_lower_triangle = torch.zeros_like(matrix)
        for i in range(1, n_row):
            for j in range(i):
                ref_lower_triangle[i, j] = -slope * (i - j)

        # Check if the lower triangle is decreasing by a constant slope (towards the bottom left corner).
        assert torch.equal(
            torch.tril(matrix, diagonal=-1), torch.tril(ref_lower_triangle, diagonal=-1)
        )


def test_rotary_attribute_access():
    cfg = HookedTransformerConfig(
        n_layers=12,
        d_model=512,
        n_ctx=1024,
        d_head=64,
        n_heads=8,
        load_in_4bit=False,
        dtype=torch.float32,
        act_fn="relu",
        rotary_dim=64,
        rotary_base=10000,
        rotary_adjacent_pairs=True,
    )

    rotary_module = RotaryEmbedding(cfg)

    class DummyAttention(AbstractAttention):
        def __init__(self):
            super().__init__(cfg)
            self.rotary_module = rotary_module

    attention = DummyAttention()

    assert torch.equal(attention.rotary_sin, rotary_module.rotary_sin), "rotary_sin does not match!"
    assert torch.equal(attention.rotary_cos, rotary_module.rotary_cos), "rotary_cos does not match!"
