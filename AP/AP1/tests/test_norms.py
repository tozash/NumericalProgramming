"""Unit tests for norm functions and reshaping."""

import numpy as np
import pytest

from src.norms import norm1_vec, norminf_vec, norm1_mat, norminf_mat
from src.reshape import vec4_to_mat2x2, mat2x2_to_vec4
from src.distances import vector_distance, matrix_distance


def test_vector_norm1():
    """Test vector 1-norm."""
    x = np.array([1, -2, 0, 3])
    assert norm1_vec(x) == 6.0


def test_vector_norminf():
    """Test vector infinity-norm."""
    x = np.array([1, -2, 0, 3])
    assert norminf_vec(x) == 3.0


def test_matrix_norm1():
    """Test matrix 1-norm (max column sum)."""
    A = np.array([[2, -1], [3, 4]])
    # Column sums: |2|+|3| = 5, |-1|+|4| = 5
    # Max = 5
    assert norm1_mat(A) == 5.0


def test_matrix_norminf():
    """Test matrix infinity-norm (max row sum)."""
    A = np.array([[2, -1], [3, 4]])
    # Row sums: |2|+|-1| = 3, |3|+|4| = 7
    # Max = 7
    assert norminf_mat(A) == 7.0


def test_vector_shape_validation():
    """Test that vector norms validate input shape."""
    with pytest.raises(ValueError):
        norm1_vec(np.array([1, 2, 3]))  # Wrong shape
    with pytest.raises(ValueError):
        norminf_vec(np.array([[1, 2], [3, 4]]))  # Wrong shape


def test_matrix_shape_validation():
    """Test that matrix norms validate input shape."""
    with pytest.raises(ValueError):
        norm1_mat(np.array([1, 2, 3, 4]))  # Wrong shape
    with pytest.raises(ValueError):
        norminf_mat(np.array([[1, 2, 3], [4, 5, 6]]))  # Wrong shape


def test_reshape_row_major():
    """Test row-major reshaping."""
    v = np.array([1, 2, 3, 4])
    M = vec4_to_mat2x2(v, order="row")
    expected = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(M, expected)

    # Round-trip
    v_back = mat2x2_to_vec4(M, order="row")
    np.testing.assert_array_equal(v_back, v)


def test_reshape_column_major():
    """Test column-major reshaping."""
    v = np.array([1, 2, 3, 4])
    M = vec4_to_mat2x2(v, order="column")
    expected = np.array([[1, 3], [2, 4]])
    np.testing.assert_array_equal(M, expected)

    # Round-trip
    v_back = mat2x2_to_vec4(M, order="column")
    np.testing.assert_array_equal(v_back, v)


def test_round_trip_random():
    """Test round-trip reshaping with random data."""
    rng = np.random.default_rng(42)
    for _ in range(10):
        v = rng.integers(-10, 10, size=4)
        M = vec4_to_mat2x2(v, order="row")
        v_back = mat2x2_to_vec4(M, order="row")
        np.testing.assert_array_equal(v_back, v)

        M_col = vec4_to_mat2x2(v, order="column")
        v_back_col = mat2x2_to_vec4(M_col, order="column")
        np.testing.assert_array_equal(v_back_col, v)


def test_vector_distance():
    """Test vector distance calculation."""
    x = np.array([1, 0, 0, 0])
    y = np.array([0, 1, 0, 0])
    assert vector_distance(x, y, "1") == 2.0
    assert vector_distance(x, y, "inf") == 1.0


def test_matrix_distance():
    """Test matrix distance calculation."""
    X = np.array([[1, 0], [0, 0]])
    Y = np.array([[0, 0], [0, 1]])
    # X - Y = [[1, 0], [0, -1]]
    # ‖X - Y‖₁ = max(|1|+|0|, |0|+|-1|) = max(1, 1) = 1
    # ‖X - Y‖∞ = max(|1|+|0|, |0|+|-1|) = max(1, 1) = 1
    assert matrix_distance(X, Y, "1") == 1.0
    assert matrix_distance(X, Y, "inf") == 1.0

