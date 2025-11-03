"""Distance calculations using vector and matrix norms."""

import numpy as np
from typing import Literal

from .norms import norm1_vec, norminf_vec, norm1_mat, norminf_mat


def vector_distance(
    x: np.ndarray, y: np.ndarray, which: Literal["1", "inf"]
) -> float:
    """
    Compute the distance between two vectors using the specified norm.

    Formula: d(x, y) = ‖x - y‖

    Args:
        x: First vector, shape (4,)
        y: Second vector, shape (4,)
        which: Norm type, either "1" or "inf"

    Returns:
        Distance between x and y

    Raises:
        ValueError: If which is not "1" or "inf"
    """
    diff = x - y
    if which == "1":
        return norm1_vec(diff)
    elif which == "inf":
        return norminf_vec(diff)
    else:
        raise ValueError(f"Invalid norm type: {which}. Must be '1' or 'inf'.")


def matrix_distance(
    X: np.ndarray, Y: np.ndarray, which: Literal["1", "inf"]
) -> float:
    """
    Compute the distance between two matrices using the specified induced norm.

    Formula: d(X, Y) = ‖X - Y‖ (induced)

    Args:
        X: First matrix, shape (2, 2)
        Y: Second matrix, shape (2, 2)
        which: Norm type, either "1" or "inf" (both are induced matrix norms)

    Returns:
        Distance between X and Y

    Raises:
        ValueError: If which is not "1" or "inf"
    """
    diff = X - Y
    if which == "1":
        return norm1_mat(diff)
    elif which == "inf":
        return norminf_mat(diff)
    else:
        raise ValueError(f"Invalid norm type: {which}. Must be '1' or 'inf'.")

