"""Functions for reshaping between 4-vectors and 2×2 matrices."""

import numpy as np
from typing import Literal


def vec4_to_mat2x2(v: np.ndarray, order: Literal["row", "column"] = "row") -> np.ndarray:
    """
    Reshape a 4-vector into a 2×2 matrix.

    Row-major order: [v₀, v₁; v₂, v₃]
        [v0  v1]
        [v2  v3]

    Column-major order: [v₀, v₂; v₁, v₃]
        [v0  v2]
        [v1  v3]

    Args:
        v: Input vector, shape (4,)
        order: Reshaping order, "row" or "column" (default: "row")

    Returns:
        2×2 matrix

    Raises:
        ValueError: If v is not a 1D array of shape (4,)
    """
    if v.shape != (4,):
        raise ValueError(f"Expected vector of shape (4,), got {v.shape}")

    if order == "row":
        return np.array([[v[0], v[1]], [v[2], v[3]]])
    elif order == "column":
        return np.array([[v[0], v[2]], [v[1], v[3]]])
    else:
        raise ValueError(f"Invalid order: {order}. Must be 'row' or 'column'.")


def mat2x2_to_vec4(M: np.ndarray, order: Literal["row", "column"] = "row") -> np.ndarray:
    """
    Reshape a 2×2 matrix into a 4-vector.

    Row-major order: [M₀₀, M₀₁, M₁₀, M₁₁]
    Column-major order: [M₀₀, M₁₀, M₀₁, M₁₁]

    Args:
        M: Input matrix, shape (2, 2)
        order: Reshaping order, "row" or "column" (default: "row")

    Returns:
        4-vector

    Raises:
        ValueError: If M is not a 2D array of shape (2, 2)
    """
    if M.shape != (2, 2):
        raise ValueError(f"Expected matrix of shape (2, 2), got {M.shape}")

    if order == "row":
        return np.array([M[0, 0], M[0, 1], M[1, 0], M[1, 1]])
    elif order == "column":
        return np.array([M[0, 0], M[1, 0], M[0, 1], M[1, 1]])
    else:
        raise ValueError(f"Invalid order: {order}. Must be 'row' or 'column'.")

