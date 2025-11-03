"""Vector and matrix norm implementations."""

import numpy as np


def norm1_vec(x: np.ndarray) -> float:
    """
    Compute the 1-norm (Manhattan norm) of a vector.

    Formula: ‖x‖₁ = ∑ᵢ |xᵢ|

    Args:
        x: Input vector, must be shape (4,)

    Returns:
        The 1-norm value as a float

    Raises:
        ValueError: If x is not a 1D array of shape (4,)
    """
    if x.shape != (4,):
        raise ValueError(f"Expected vector of shape (4,), got {x.shape}")
    return float(np.sum(np.abs(x)))


def norminf_vec(x: np.ndarray) -> float:
    """
    Compute the infinity-norm (maximum norm) of a vector.

    Formula: ‖x‖∞ = maxᵢ |xᵢ|

    Args:
        x: Input vector, must be shape (4,)

    Returns:
        The infinity-norm value as a float

    Raises:
        ValueError: If x is not a 1D array of shape (4,)
    """
    if x.shape != (4,):
        raise ValueError(f"Expected vector of shape (4,), got {x.shape}")
    return float(np.max(np.abs(x)))


def norm1_mat(A: np.ndarray) -> float:
    """
    Compute the 1-norm (induced) of a matrix.

    Formula: ‖A‖₁ = maxⱼ ∑ᵢ |aᵢⱼ| (maximum column sum)

    This is the matrix norm induced by the vector 1-norm.

    Args:
        A: Input matrix, must be shape (2, 2)

    Returns:
        The 1-norm value as a float

    Raises:
        ValueError: If A is not a 2D array of shape (2, 2)
    """
    if A.shape != (2, 2):
        raise ValueError(f"Expected matrix of shape (2, 2), got {A.shape}")
    # Sum absolute values along rows (axis 0), then take maximum
    column_sums = np.sum(np.abs(A), axis=0)
    return float(np.max(column_sums))


def norminf_mat(A: np.ndarray) -> float:
    """
    Compute the infinity-norm (induced) of a matrix.

    Formula: ‖A‖∞ = maxᵢ ∑ⱼ |aᵢⱼ| (maximum row sum)

    This is the matrix norm induced by the vector infinity-norm.

    Args:
        A: Input matrix, must be shape (2, 2)

    Returns:
        The infinity-norm value as a float

    Raises:
        ValueError: If A is not a 2D array of shape (2, 2)
    """
    if A.shape != (2, 2):
        raise ValueError(f"Expected matrix of shape (2, 2), got {A.shape}")
    # Sum absolute values along columns (axis 1), then take maximum
    row_sums = np.sum(np.abs(A), axis=1)
    return float(np.max(row_sums))

