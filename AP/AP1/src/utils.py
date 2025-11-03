"""Utility functions for random number generation."""

import numpy as np
from numpy.random import Generator


def rng(seed: int = 0) -> Generator:
    """
    Create a deterministic random number generator.

    Args:
        seed: Random seed (default: 0)

    Returns:
        NumPy random number generator instance
    """
    return np.random.default_rng(seed)


def two_random_vec4s(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two random 4-vectors with small integer values.

    Args:
        seed: Random seed (default: 0)

    Returns:
        Tuple of two 4-vectors with values in range [-3, 3]
    """
    gen = rng(seed)
    x = gen.integers(-3, 4, size=4, dtype=np.int64)
    y = gen.integers(-3, 4, size=4, dtype=np.int64)
    return x, y

