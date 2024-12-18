"""matrix.

This module contains utility functions related to the transformer lens implementation of factored
matrices.
"""
import torch
from typing import Tuple, Union, cast
from jaxtyping import Float, Int
from transformer_lens.FactoredMatrix import FactoredMatrix


def get_corner(tensor, n=3):
    # Prints the top left corner of the tensor
    if isinstance(tensor, torch.Tensor):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))]
    elif isinstance(tensor, FactoredMatrix):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))].AB
    
def composition_scores(
    left: FactoredMatrix, right: FactoredMatrix, broadcast_dims=True
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