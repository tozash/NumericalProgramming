"""Unit ball visualization for vectors and matrices."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Literal, Optional, Tuple

from .norms import norm1_vec, norminf_vec, norm1_mat, norminf_mat
from .reshape import vec4_to_mat2x2, mat2x2_to_vec4


def plot_vector_unit_ball_slice(
    norm_kind: Literal["1", "inf"],
    xr: np.ndarray,
    vary_idx: Tuple[int, int] = (0, 1),
    fixed: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """
    Plot a 2D slice of the unit ball centered at xr.

    The unit ball is defined as {z : ‖z - xr‖ ≤ 1}, projected onto
    the plane spanned by coordinates vary_idx.

    Args:
        norm_kind: Norm type, "1" or "inf"
        xr: Reference vector of shape (4,)
        vary_idx: Tuple of two indices to vary (default: (0, 1))
        fixed: Fixed values for other coordinates (default: xr values)
        ax: Matplotlib axes (default: creates new figure)

    Returns:
        Tuple of (figure, axes) objects

    Raises:
        ValueError: If norm_kind is not "1" or "inf"
        ValueError: If xr is not shape (4,)
    """
    if xr.shape != (4,):
        raise ValueError(f"Expected reference vector of shape (4,), got {xr.shape}")
    if norm_kind not in ("1", "inf"):
        raise ValueError(f"Invalid norm_kind: {norm_kind}. Must be '1' or 'inf'.")

    if fixed is None:
        fixed = xr.copy()

    i, j = vary_idx
    other_indices = [idx for idx in range(4) if idx not in vary_idx]

    # Create grid in the (i, j) plane
    grid_size = 201
    extent = 3.0  # Extent around reference
    u = np.linspace(xr[i] - extent, xr[i] + extent, grid_size)
    v = np.linspace(xr[j] - extent, xr[j] + extent, grid_size)
    U, V = np.meshgrid(u, v)

    # Evaluate membership: ‖z - xr‖ ≤ 1
    membership = np.zeros_like(U, dtype=bool)
    for idx_u in range(grid_size):
        for idx_v in range(grid_size):
            z = xr.copy()
            z[i] = U[idx_v, idx_u]
            z[j] = V[idx_v, idx_u]
            # Set other coordinates to fixed values
            for other_idx in other_indices:
                z[other_idx] = fixed[other_idx]

            diff = z - xr
            if norm_kind == "1":
                dist = norm1_vec(diff)
            else:  # norm_kind == "inf"
                dist = norminf_vec(diff)

            membership[idx_v, idx_u] = dist <= 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Plot filled contour
    ax.contourf(U, V, membership.astype(float), levels=[0, 0.5, 1], colors=['white', 'lightblue'], alpha=0.3)
    ax.contour(U, V, membership.astype(float), levels=[0.5], colors=['blue'], linewidths=2)

    ax.set_xlabel(f"Coordinate {i}")
    ax.set_ylabel(f"Coordinate {j}")
    norm_name = "∞" if norm_kind == "inf" else "1"
    ax.set_title(f"Vector Unit Ball Slice (‖·‖_{norm_name}), Varying ({i},{j})")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig, ax


def plot_matrix_unit_ball_slice(
    norm_kind: Literal["1", "inf"],
    Xr: np.ndarray,
    vary_idx: Tuple[int, int] = (0, 1),
    order: Literal["row", "column"] = "row",
    fixed: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
) -> tuple[Figure, Axes]:
    """
    Plot a 2D slice of the matrix unit ball centered at Xr.

    The unit ball is defined as {Z : ‖Z - Xr‖ ≤ 1} (using induced norm),
    projected onto the plane spanned by two vector coordinates (interpreted
    in the chosen order).

    Args:
        norm_kind: Norm type, "1" or "inf" (induced matrix norm)
        Xr: Reference matrix of shape (2, 2)
        vary_idx: Tuple of two vector indices to vary (default: (0, 1))
        order: Reshaping order for vector interpretation (default: "row")
        fixed: Fixed values for other coordinates (default: Xr values)
        ax: Matplotlib axes (default: creates new figure)

    Returns:
        Tuple of (figure, axes) objects

    Raises:
        ValueError: If Xr is not shape (2, 2)
        ValueError: If norm_kind is not "1" or "inf"
    """
    if Xr.shape != (2, 2):
        raise ValueError(f"Expected reference matrix of shape (2, 2), got {Xr.shape}")
    if norm_kind not in ("1", "inf"):
        raise ValueError(f"Invalid norm_kind: {norm_kind}. Must be '1' or 'inf'.")

    # Convert matrix to 4-vector using reshape function
    xr_vec = mat2x2_to_vec4(Xr, order=order)

    if fixed is None:
        fixed = xr_vec.copy()

    i, j = vary_idx
    other_indices = [idx for idx in range(4) if idx not in vary_idx]

    # Create grid in the (i, j) plane
    grid_size = 201
    extent = 3.0
    u = np.linspace(xr_vec[i] - extent, xr_vec[i] + extent, grid_size)
    v = np.linspace(xr_vec[j] - extent, xr_vec[j] + extent, grid_size)
    U, V = np.meshgrid(u, v)

    # Evaluate membership: ‖Z - Xr‖ ≤ 1 (using induced norm)
    membership = np.zeros_like(U, dtype=bool)
    for idx_u in range(grid_size):
        for idx_v in range(grid_size):
            # Construct 4-vector
            z_vec = xr_vec.copy()
            z_vec[i] = U[idx_v, idx_u]
            z_vec[j] = V[idx_v, idx_u]
            for other_idx in other_indices:
                z_vec[other_idx] = fixed[other_idx]

            # Convert to 2×2 matrix using reshape function
            Z = vec4_to_mat2x2(z_vec, order=order)

            diff = Z - Xr
            if norm_kind == "1":
                dist = norm1_mat(diff)
            else:  # norm_kind == "inf"
                dist = norminf_mat(diff)

            membership[idx_v, idx_u] = dist <= 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Plot filled contour
    ax.contourf(U, V, membership.astype(float), levels=[0, 0.5, 1], colors=['white', 'lightcoral'], alpha=0.3)
    ax.contour(U, V, membership.astype(float), levels=[0.5], colors=['red'], linewidths=2)

    ax.set_xlabel(f"Vector Coordinate {i} (in {order} order)")
    ax.set_ylabel(f"Vector Coordinate {j} (in {order} order)")
    norm_name = "∞" if norm_kind == "inf" else "1"
    ax.set_title(f"Matrix Unit Ball Slice (‖·‖_{norm_name}, induced), Varying ({i},{j})")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    return fig, ax

